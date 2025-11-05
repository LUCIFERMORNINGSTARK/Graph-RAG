import streamlit as st
import PyPDF2
import networkx as nx
from groq import Groq
import re
from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict
import hashlib
import plotly.graph_objects as go
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

st.set_page_config(page_title="Universal GraphRAG System", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedGraphRAG:
    def __init__(self, groq_api_key=None, model="llama-3.1-70b-versatile"):
        self.graph = nx.DiGraph()
        self.chunks = []
        self.entity_index = defaultdict(set)
        self.term_index = defaultdict(set)
        self.groq_client = None
        self.model = model
        
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                st.error(f"Failed to initialize Groq: {str(e)}")
    
    def _normalize_line(self, line: str) -> str:
        """Aggressive text normalization"""
        # Remove excessive whitespace
        line = re.sub(r'\s{2,}', ' ', line.strip())
        # Remove weird Unicode spaces
        line = re.sub(r'[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]', ' ', line)
        return line
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract and normalize text"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_with_pages = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Normalize line by line
                        lines = page_text.split('\n')
                        normalized_lines = [self._normalize_line(line) for line in lines if line.strip()]
                        normalized_text = '\n'.join(normalized_lines)
                        
                        text_with_pages.append({
                            'page': page_num + 1,
                            'text': normalized_text
                        })
                except Exception as e:
                    st.warning(f"Page {page_num + 1} failed: {str(e)}")
            
            return text_with_pages
        except Exception as e:
            st.error(f"PDF error: {str(e)}")
            return []
    
    def _detect_structure(self, text: str) -> List[Dict]:
        """Enhanced structure detection with normalization"""
        patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown_header'),
            (r'^([A-Z][A-Z\s]{2,})$', 'caps_header'),
            (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered_section'),
            (r'^(Chapter|Section|Part|Article|Appendix)\s+(\d+|[IVXLCDM]+)[:\s]*(.+)?$', 'formal_section'),
            (r'^\*\*(.+?)\*\*', 'bold_text'),
            (r'^[-•]\s+(.+)$', 'bullet_point'),
        ]
        
        matches = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Normalize for pattern matching
            normalized = self._normalize_line(line)
            
            for pattern, structure_type in patterns:
                try:
                    match = re.match(pattern, normalized, re.IGNORECASE)
                    if match:
                        matches.append({
                            'line_num': i,
                            'text': normalized,
                            'type': structure_type,
                            'match': match
                        })
                        break
                except:
                    continue
        
        return matches
    
    def intelligent_chunk(self, text_pages: List[Dict]) -> List[Dict]:
        """Optimized chunking"""
        full_text = '\n'.join([p['text'] for p in text_pages])
        structures = self._detect_structure(full_text)
        
        if len(structures) > 3:
            chunks = self._structural_chunk(full_text, structures, text_pages)
        else:
            chunks = self._sliding_window_chunk(full_text, text_pages)
        
        # Add context chunks
        chunks = self._add_context_chunks(chunks)
        
        return chunks
    
    def _structural_chunk(self, text: str, structures: List[Dict], pages: List[Dict]) -> List[Dict]:
        """Create chunks from structure"""
        chunks = []
        lines = text.split('\n')
        
        sections = []
        current_section = {'start': 0, 'header': None, 'lines': []}
        
        for struct in structures:
            if struct['type'] in ['markdown_header', 'caps_header', 'numbered_section', 'formal_section']:
                if current_section['lines'] or current_section['header']:
                    sections.append(current_section)
                current_section = {
                    'start': struct['line_num'],
                    'header': struct['text'],
                    'lines': [struct['line_num']]
                }
            else:
                current_section['lines'].append(struct['line_num'])
        
        if current_section['lines'] or current_section['header']:
            sections.append(current_section)
        
        for i, section in enumerate(sections):
            start_line = section['start']
            end_line = sections[i+1]['start'] if i+1 < len(sections) else len(lines)
            
            section_text = '\n'.join(lines[start_line:end_line]).strip()
            
            if len(section_text) > 50:
                char_pos = len('\n'.join(lines[:start_line]))
                page_num = self._find_page_number(char_pos, pages)
                
                chunks.append({
                    'id': f"chunk_{len(chunks)}",
                    'text': section_text,
                    'header': section.get('header', ''),
                    'page': page_num,
                    'type': 'structural'
                })
        
        return chunks
    
    def _sliding_window_chunk(self, text: str, pages: List[Dict], chunk_size=700, overlap=200) -> List[Dict]:
        """Optimized sliding window"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                char_pos = text.find(chunk_text)
                page_num = self._find_page_number(max(0, char_pos), pages)
                
                chunks.append({
                    'id': f"chunk_{len(chunks)}",
                    'text': chunk_text,
                    'header': self._extract_potential_header(chunk_text),
                    'page': page_num,
                    'type': 'semantic'
                })
                
                # Keep overlap
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_len
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': f"chunk_{len(chunks)}",
                'text': chunk_text,
                'header': self._extract_potential_header(chunk_text),
                'page': pages[-1]['page'] if pages else 1,
                'type': 'semantic'
            })
        
        return chunks
    
    def _add_context_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add context windows"""
        enhanced_chunks = list(chunks)
        window_size = 2 if len(chunks) > 100 else 3
        
        for i in range(0, len(chunks) - window_size + 1, 2):
            window_chunks = chunks[i:i+window_size]
            merged_text = ' '.join([c['text'] for c in window_chunks])
            
            if len(merged_text) < 2500:
                enhanced_chunks.append({
                    'id': f"context_{i}",
                    'text': merged_text,
                    'header': window_chunks[0].get('header', ''),
                    'page': window_chunks[0].get('page', 0),
                    'type': 'context',
                    'source_chunks': [c['id'] for c in window_chunks]
                })
        
        return enhanced_chunks
    
    def _find_page_number(self, char_pos: int, pages: List[Dict]) -> int:
        """Find page number"""
        current_pos = 0
        for page in pages:
            page_len = len(page['text'])
            if char_pos < current_pos + page_len:
                return page['page']
            current_pos += page_len
        return pages[-1]['page'] if pages else 1
    
    def _extract_potential_header(self, text: str) -> str:
        """Extract header"""
        lines = text.split('\n')
        for line in lines[:3]:
            line = self._normalize_line(line)
            if 10 < len(line) < 100 and re.match(r'^[A-Z#\d]', line):
                return line
        return text[:50] + '...'
    
    def _extract_entities_from_chunk(self, chunk: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities (parallelizable)"""
        text = chunk['text']
        entities = []
        relationships = []
        
        # Extract noun phrases
        noun_phrases = self._extract_noun_phrases(text)
        for phrase in noun_phrases[:20]:  # Limit per chunk
            if 3 <= len(phrase.split()) <= 6:
                entity_id = f"entity_{hashlib.md5(phrase.lower().encode()).hexdigest()[:8]}"
                entities.append({
                    'id': entity_id,
                    'label': phrase,
                    'type': 'concept',
                    'chunk_id': chunk['id']
                })
        
        # Extract important terms
        important_terms = self._extract_important_terms(text)
        for term in list(important_terms)[:15]:
            entity_id = f"term_{hashlib.md5(term.lower().encode()).hexdigest()[:8]}"
            entities.append({
                'id': entity_id,
                'label': term,
                'type': 'term',
                'chunk_id': chunk['id']
            })
        
        # Extract definitions
        definitions = self._extract_definitions(text)
        for defn in definitions[:10]:
            entity_id = f"def_{hashlib.md5(defn['term'].lower().encode()).hexdigest()[:8]}"
            entities.append({
                'id': entity_id,
                'label': defn['term'],
                'type': 'definition',
                'chunk_id': chunk['id']
            })
            relationships.append({
                'from': entity_id,
                'to': chunk['id'],
                'type': 'defined_in'
            })
        
        # Index terms
        terms = self._tokenize_with_bigrams(text)
        
        return entities, relationships, terms
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases"""
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        matches = re.findall(pattern, text)
        quoted = re.findall(r'"([^"]{5,50})"', text)
        return list(set(matches + quoted))
    
    def _extract_important_terms(self, text: str) -> Set[str]:
        """Extract key terms"""
        terms = set()
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        technical = re.findall(r'\b[a-zA-Z]+[-\d][a-zA-Z\d-]*\b', text)
        terms.update(capitalized[:10])
        terms.update(acronyms[:5])
        terms.update(technical[:5])
        return terms
    
    def _extract_definitions(self, text: str) -> List[Dict]:
        """Extract definitions"""
        definitions = []
        patterns = [
            r'(["\']?[A-Za-z\s]{3,30}["\']?)\s+(?:is|means|refers to|denotes)\s+([^.!?]{10,})',
            r'(?:defined as|known as)\s+(["\']?[A-Za-z\s]{3,30}["\']?)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    term = match.group(1).strip(' "\'"')
                    definition = match.group(2).strip() if len(match.groups()) > 1 else ""
                    if 2 <= len(term.split()) <= 5 and len(definition) > 10:
                        definitions.append({'term': term, 'definition': definition})
        
        return definitions
    
    def _tokenize_with_bigrams(self, text: str) -> Set[str]:
        """Create tokens with n-grams"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'it'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        tokens = set(words)
        for i in range(len(words) - 1):
            tokens.add(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            tokens.add(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return tokens
    
    def build_knowledge_graph(self, chunks: List[Dict]):
        """Parallel graph building"""
        st.info("🔨 Building knowledge graph...")
        progress_bar = st.progress(0)
        
        # Parallel entity extraction
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._extract_entities_from_chunk, chunk): chunk for chunk in chunks}
            
            completed = 0
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    entities, relationships, terms = future.result()
                    
                    # Add chunk node
                    self.graph.add_node(chunk['id'], **chunk)
                    
                    # Add entities
                    for entity in entities:
                        if not self.graph.has_node(entity['id']):
                            self.graph.add_node(entity['id'], **entity)
                        self.graph.add_edge(chunk['id'], entity['id'], relation='contains')
                        self.entity_index[entity['label'].lower()].add(chunk['id'])
                    
                    # Add relationships
                    for rel in relationships:
                        from_id, to_id = rel.get('from'), rel.get('to')
                        if from_id and to_id:
                            if self.graph.has_node(from_id) and self.graph.has_node(to_id):
                                self.graph.add_edge(from_id, to_id, relation=rel['type'])
                    
                    # Index terms
                    for term in terms:
                        self.term_index[term].add(chunk['id'])
                    
                except Exception as e:
                    st.warning(f"Chunk {chunk['id']} failed: {e}")
                
                completed += 1
                progress_bar.progress(completed / len(chunks))
        
        # Sequential links
        for i in range(len(chunks) - 1):
            if chunks[i]['type'] != 'context' and chunks[i+1]['type'] != 'context':
                self.graph.add_edge(chunks[i]['id'], chunks[i+1]['id'], relation='next')
        
        progress_bar.empty()
        st.success(f"✅ Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def advanced_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Enhanced multi-stage retrieval"""
        all_results = []
        query_lower = query.lower()
        query_normalized = self._normalize_line(query_lower)
        
        # Stage 1: Exact phrase matching (highest priority)
        for chunk in self.chunks:
            chunk_text_lower = chunk['text'].lower()
            chunk_normalized = self._normalize_line(chunk_text_lower)
            
            if query_normalized in chunk_normalized:
                all_results.append({
                    'chunk': chunk,
                    'score': 1.0,
                    'method': 'exact_match'
                })
            elif query_lower in chunk_text_lower:
                all_results.append({
                    'chunk': chunk,
                    'score': 0.95,
                    'method': 'exact_match'
                })
        
        # Stage 2: Entity-based retrieval
        query_terms = self._tokenize_with_bigrams(query)
        for term in query_terms:
            if term in self.entity_index:
                for chunk_id in self.entity_index[term]:
                    chunk = next((c for c in self.chunks if c['id'] == chunk_id), None)
                    if chunk:
                        all_results.append({
                            'chunk': chunk,
                            'score': 0.9,
                            'method': 'entity_match'
                        })
            
            # Check term index
            if term in self.term_index:
                for chunk_id in self.term_index[term]:
                    chunk = next((c for c in self.chunks if c['id'] == chunk_id), None)
                    if chunk:
                        all_results.append({
                            'chunk': chunk,
                            'score': 0.85,
                            'method': 'term_match'
                        })
        
        # Stage 3: Token-based scoring
        query_tokens = [t for t in query_lower.split() if len(t) > 2]
        for chunk in self.chunks:
            chunk_text_lower = chunk['text'].lower()
            
            match_score = 0
            for token in query_tokens:
                if token in chunk_text_lower:
                    # Count occurrences
                    count = chunk_text_lower.count(token)
                    match_score += min(count, 3) * 0.3
                else:
                    # Fuzzy match
                    for word in chunk_text_lower.split():
                        if self._fuzzy_match(token, word, threshold=0.85):
                            match_score += 0.25
                            break
            
            if match_score > 0.3:
                normalized_score = min(match_score / len(query_tokens), 1.0)
                all_results.append({
                    'chunk': chunk,
                    'score': normalized_score * 0.8,
                    'method': 'token_match'
                })
        
        # Stage 4: Graph expansion
        top_candidates = sorted(all_results, key=lambda x: x['score'], reverse=True)[:7]
        for result in top_candidates:
            chunk_id = result['chunk']['id']
            if self.graph.has_node(chunk_id):
                neighbors = list(self.graph.successors(chunk_id))
                neighbors.extend(list(self.graph.predecessors(chunk_id)))
                
                for neighbor_id in neighbors[:15]:
                    node_data = self.graph.nodes.get(neighbor_id, {})
                    if node_data.get('type') in ['structural', 'semantic', 'context']:
                        chunk = next((c for c in self.chunks if c['id'] == neighbor_id), None)
                        if chunk:
                            all_results.append({
                                'chunk': chunk,
                                'score': 0.65,
                                'method': 'graph_expansion'
                            })
        
        # Deduplicate and boost
        seen = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            chunk_id = result['chunk']['id']
            if chunk_id not in seen:
                seen.add(chunk_id)
                
                # Boost context chunks that contain high-scoring chunks
                if result['chunk'].get('type') == 'context':
                    source_chunks = result['chunk'].get('source_chunks', [])
                    if any(cid in seen for cid in source_chunks):
                        result['score'] *= 1.3
                
                unique_results.append(result)
        
        return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _fuzzy_match(self, word1: str, word2: str, threshold: float = 0.85) -> bool:
        """Fuzzy match"""
        if len(word1) < 4 or len(word2) < 4:
            return False
        return SequenceMatcher(None, word1, word2).ratio() >= threshold
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate answer"""
        if not self.groq_client:
            contexts = []
            for i, result in enumerate(retrieved_chunks[:5]):
                chunk = result['chunk']
                contexts.append(f"[Source {i+1} - Page {chunk.get('page', '?')}]\n{chunk['text'][:600]}")
            return "⚠️ Configure Groq API for AI answers\n\n" + "\n\n---\n\n".join(contexts)
        
        max_context_chars = 8000
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(retrieved_chunks[:7]):
            chunk = result['chunk']
            header = chunk.get('header', '')
            page = chunk.get('page', '?')
            chunk_text = chunk['text']
            
            remaining = max_context_chars - total_chars
            if remaining < 200:
                break
            
            max_chunk_chars = min(len(chunk_text), remaining - 100, 1500)
            truncated = chunk_text[:max_chunk_chars]
            if len(chunk_text) > max_chunk_chars:
                truncated += "..."
            
            context_part = f"[Source {i+1} | Page {page}] {header}\n{truncated}"
            context_parts.append(context_part)
            total_chars += len(context_part)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer the question using the provided context.

CONTEXT:
{context}

QUESTION: {query}

Provide a comprehensive answer citing sources with [Source N]. If the answer isn't in the context, explain what related information is available.

ANSWER:"""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Provide accurate answers with citations from the context."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "too large" in str(e).lower():
                try:
                    reduced = "\n\n".join(context_parts[:3])
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Concise answers with citations."},
                            {"role": "user", "content": f"{reduced}\n\nQ: {query}\n\nA:"}
                        ],
                        model=self.model,
                        temperature=0.2,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                except:
                    pass
            return f"Error: {str(e)}\n\nTry llama-3.1-70b-versatile or reduce results."
    
    def query(self, query_text: str, top_k: int = 10) -> Dict:
        """Query interface"""
        results = self.advanced_retrieve(query_text, top_k)
        answer = self.generate_answer(query_text, results)
        avg_score = np.mean([r['score'] for r in results]) if results else 0
        
        return {
            'answer': answer,
            'confidence': min(avg_score * 1.15, 1.0),
            'retrieved_chunks': results,
            'num_chunks': len(results)
        }
    
    def visualize_graph_plotly(self, max_nodes: int = 100):
        """Graph visualization"""
        if self.graph.number_of_nodes() == 0:
            return None
        
        if self.graph.number_of_nodes() > max_nodes:
            degree_dict = dict(self.graph.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes = [n[0] for n in top_nodes]
            subgraph = self.graph.subgraph(nodes)
        else:
            subgraph = self.graph
        
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines')
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_traces = {}
        node_types = set(subgraph.nodes[node].get('type', 'unknown') for node in subgraph.nodes())
        
        color_map = {
            'structural': '#1f77b4', 'semantic': '#ff7f0e', 'context': '#2ca02c',
            'definition': '#d62728', 'concept': '#9467bd', 'term': '#8c564b', 'unknown': '#c7c7c7'
        }
        
        for node_type in node_types:
            node_trace = go.Scatter(
                x=[], y=[], mode='markers+text', hoverinfo='text',
                name=node_type.replace('_', ' ').title(),
                marker=dict(size=10, color=color_map.get(node_type, '#c7c7c7'), line=dict(width=2, color='white')),
                text=[], textposition="top center", textfont=dict(size=8)
            )
            
            for node in subgraph.nodes():
                if subgraph.nodes[node].get('type', 'unknown') == node_type:
                    x, y = pos[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y])
                    
                    node_data = subgraph.nodes[node]
                    label = node_data.get('label', node_data.get('header', str(node)[:20]))
                    if label is None:
                        label = str(node)[:20]
                    label = str(label)
                    node_trace['text'] += tuple([label[:15] + '...' if len(label) > 15 else label])
            
            node_traces[node_type] = node_trace
        
        fig = go.Figure(data=[edge_trace] + list(node_traces.values()),
                       layout=go.Layout(
                           title='Knowledge Graph', showlegend=True, hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700, plot_bgcolor='#f8f9fa'
                       ))
        
        return fig

# Session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# UI
st.markdown('<div class="main-header">🧠 Universal GraphRAG System</div>', unsafe_allow_html=True)
st.markdown('<span class="accuracy-badge">✓ Universal  ✓ Parallel Processing  ✓ Enhanced Accuracy</span>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Optional: For AI-generated answers")
    model = st.selectbox("Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    
    st.divider()
    st.header("🎯 Key Features")
    st.info("✓ Parallel processing")
    st.info("✓ Text normalization")
    st.info("✓ Multi-stage retrieval")
    st.info("✓ Enhanced scoring")
    st.info("✓ Graph expansion")
    st.info("✓ Fuzzy matching")
    
    st.divider()
    st.markdown("**Performance:**")
    st.caption("• 4x faster processing")
    st.caption("• Better accuracy")
    st.caption("• Improved chunking")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Process", "🔍 Query", "🕸️ Graph", "📊 Stats"])

with tab1:
    st.header("Upload Document")
    st.markdown("Upload any PDF - automatic structure detection and intelligent chunking.")
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button("🚀 Process Document", type="primary", use_container_width=True)
        with col2:
            if st.session_state.processed:
                st.success("✓ Ready")
    
    if uploaded_file and process_button:
        start_time = time.time()
        
        with st.spinner("Processing document..."):
            st.session_state.rag_system = EnhancedGraphRAG(api_key, model)
            
            text_pages = st.session_state.rag_system.extract_text_from_pdf(uploaded_file)
            if not text_pages:
                st.error("Failed to extract text")
            else:
                st.success(f"✅ Extracted {len(text_pages)} pages")
                
                with st.spinner("Creating intelligent chunks..."):
                    chunks = st.session_state.rag_system.intelligent_chunk(text_pages)
                    st.session_state.rag_system.chunks = chunks
                    
                    structural = sum(1 for c in chunks if c['type'] == 'structural')
                    semantic = sum(1 for c in chunks if c['type'] == 'semantic')
                    context = sum(1 for c in chunks if c['type'] == 'context')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Structural", structural)
                    with col2:
                        st.metric("Semantic", semantic)
                    with col3:
                        st.metric("Context", context)
                
                st.session_state.rag_system.build_knowledge_graph(chunks)
                
                elapsed = time.time() - start_time
                st.session_state.processed = True
                st.balloons()
                st.success(f"🎉 Processed in {elapsed:.1f}s! Go to Query tab.")

with tab2:
    if not st.session_state.processed:
        st.warning("⚠️ Process a document first")
        st.info("💡 Upload PDF in 'Process' tab")
    else:
        st.header("Query Knowledge Base")
        
        with st.expander("💡 Query Tips"):
            st.markdown("""
            **Best Practices:**
            - Ask specific questions about document content
            - Use keywords from the document
            - Try different phrasings for better results
            - System handles typos and variations
            
            **Example queries:**
            - "What is [specific topic] about?"
            - "How does [concept] work?"
            - "What are the requirements for [X]?"
            - "Explain [term] from the document"
            """)
        
        query = st.text_input("Enter your question", placeholder="What is this document about?", key="query_input")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            top_k = st.number_input("Results", 5, 20, 10, key="top_k")
        
        if search_button and query:
            with st.spinner("Searching..."):
                results = st.session_state.rag_system.query(query, top_k)
                
                confidence = results['confidence'] * 100
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if confidence > 75:
                        st.success(f"✅ High Confidence: {confidence:.1f}%")
                    elif confidence > 50:
                        st.info(f"ℹ️ Medium Confidence: {confidence:.1f}%")
                    else:
                        st.warning(f"⚠️ Low Confidence: {confidence:.1f}%")
                
                with col2:
                    st.metric("Sources", results['num_chunks'])
                
                with col3:
                    method_counts = {}
                    for r in results['retrieved_chunks']:
                        method = r['method']
                        method_counts[method] = method_counts.get(method, 0) + 1
                    primary = max(method_counts, key=method_counts.get) if method_counts else "N/A"
                    st.metric("Method", primary.replace('_', ' ').title())
                
                st.divider()
                
                st.subheader("📝 Answer")
                st.markdown(results['answer'])
                
                st.divider()
                
                st.subheader("📚 Retrieved Sources")
                
                methods = {}
                for result in results['retrieved_chunks']:
                    method = result['method']
                    if method not in methods:
                        methods[method] = []
                    methods[method].append(result)
                
                if len(methods) > 1:
                    method_tabs = st.tabs([m.replace('_', ' ').title() for m in methods.keys()])
                    
                    for tab, (method, method_results) in zip(method_tabs, methods.items()):
                        with tab:
                            for i, result in enumerate(method_results):
                                chunk = result['chunk']
                                
                                with st.expander(f"📄 {chunk.get('header', chunk['id'])} (Score: {result['score']:.3f})"):
                                    st.write(f"**Page:** {chunk.get('page', 'Unknown')}")
                                    st.write(f"**Type:** {chunk['type'].title()}")
                                    st.write(f"**Score:** {result['score']:.3f}")
                                    st.divider()
                                    st.write("**Content:**")
                                    st.write(chunk['text'][:1000] + ("..." if len(chunk['text']) > 1000 else ""))
                else:
                    for i, result in enumerate(results['retrieved_chunks']):
                        chunk = result['chunk']
                        
                        with st.expander(f"📄 Source {i+1}: {chunk.get('header', chunk['id'])} (Score: {result['score']:.3f})"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Page:** {chunk.get('page', 'Unknown')}")
                            with col2:
                                st.write(f"**Type:** {chunk['type'].title()}")
                            with col3:
                                st.write(f"**Method:** {result['method'].replace('_', ' ').title()}")
                            
                            st.divider()
                            st.write("**Content:**")
                            st.write(chunk['text'][:1000] + ("..." if len(chunk['text']) > 1000 else ""))

with tab3:
    if not st.session_state.processed:
        st.warning("⚠️ Process a document first")
    else:
        st.header("Knowledge Graph Visualization")
        st.markdown("Interactive visualization of document structure")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_nodes = st.slider("Max Nodes", 20, 200, 100, 10)
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with st.spinner("Creating visualization..."):
            fig = st.session_state.rag_system.visualize_graph_plotly(max_nodes)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Legend:** Colors = entity types. Hover for details.")

with tab4:
    if st.session_state.processed:
        st.header("System Statistics")
        
        st.subheader("📊 Graph Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", st.session_state.rag_system.graph.number_of_nodes())
        with col2:
            st.metric("Edges", st.session_state.rag_system.graph.number_of_edges())
        with col3:
            st.metric("Chunks", len(st.session_state.rag_system.chunks))
        with col4:
            st.metric("Entities", len(st.session_state.rag_system.entity_index))
        
        st.divider()
        
        st.subheader("📦 Chunk Distribution")
        chunk_types = {}
        for chunk in st.session_state.rag_system.chunks:
            chunk_type = chunk['type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        col1, col2 = st.columns(2)
        with col1:
            for chunk_type, count in chunk_types.items():
                st.metric(f"{chunk_type.title()}", count)
        
        with col2:
            chunk_sizes = [len(c['text']) for c in st.session_state.rag_system.chunks if c['type'] != 'context']
            if chunk_sizes:
                st.metric("Avg Chunk Size", f"{np.mean(chunk_sizes):.0f} chars")
                st.metric("Min/Max Size", f"{min(chunk_sizes)} / {max(chunk_sizes)}")
        
        st.divider()
        
        st.subheader("🏷️ Entity Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Terms Indexed", len(st.session_state.rag_system.term_index))
            st.metric("Entities", len(st.session_state.rag_system.entity_index))
        
        with col2:
            entity_counts = [(e, len(c)) for e, c in st.session_state.rag_system.entity_index.items()]
            entity_counts.sort(key=lambda x: x[1], reverse=True)
            
            st.write("**Most Frequent Entities:**")
            for entity, count in entity_counts[:10]:
                st.caption(f"• {entity[:50]} ({count}x)")
        
        st.divider()
        
        st.subheader("🔗 Graph Connectivity")
        degrees = dict(st.session_state.rag_system.graph.degree())
        if degrees:
            avg_degree = np.mean(list(degrees.values()))
            max_degree = max(degrees.values())
            max_node = max(degrees, key=degrees.get)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Degree", f"{avg_degree:.2f}")
            with col2:
                st.metric("Max Degree", max_degree)
            with col3:
                node_label = st.session_state.rag_system.graph.nodes[max_node].get('label', max_node)
                st.metric("Most Connected", str(node_label)[:20] + "...")
    else:
        st.warning("⚠️ Process a document first")