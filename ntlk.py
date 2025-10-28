import nltk
import spacy
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from typing import List, Dict, Tuple
import numpy as np

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class ComputerTermExtractor:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        # 扩展停用词列表，包含计算机领域常见但无意义的词
        self.extra_stop_words = {
            'using', 'use', 'used', 'show', 'shows', 'shown', 
            'propose', 'proposed', 'method', 'approach', 'paper',
            'result', 'results', 'evaluation', 'evaluate', 'performance',
            'based', 'proposed', 'novel', 'significantly', 'compared'
        }
        self.stop_words.update(self.extra_stop_words)
        
        # 加载spacy模型用于词性标注和实体识别
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("请先安装spacy模型: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        # 移除标点符号和数字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # 分词
        tokens = nltk.word_tokenize(text)
        # 移除停用词和短词
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """提取名词短语"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower()
            # 过滤短短语和包含停用词的短语
            if len(phrase.split()) >= 2 and len(phrase) > 4:
                clean_phrase = ' '.join([token for token in phrase.split() 
                                       if token not in self.stop_words])
                if len(clean_phrase) > 3:
                    noun_phrases.append(clean_phrase)
        
        return noun_phrases
    
    def extract_technical_terms(self, abstracts: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        从摘要中提取技术术语
        
        参数:
            abstracts: 摘要列表
            
        返回:
            包含不同类型术语的字典
        """
        # 预处理所有摘要
        all_tokens = []
        all_noun_phrases = []
        
        for abstract in abstracts:
            # 单字词提取
            tokens = self.preprocess_text(abstract)
            all_tokens.extend(tokens)
            
            # 名词短语提取
            noun_phrases = self.extract_noun_phrases(abstract)
            all_noun_phrases.extend(noun_phrases)
        
        # 计算单字词频率
        word_freq = Counter(all_tokens)
        
        # 计算名词短语频率
        phrase_freq = Counter(all_noun_phrases)
        
        # 使用TF-IDF提取重要词汇
        tfidf_terms = self.extract_tfidf_terms(abstracts)
        
        # 过滤和排序结果
        filtered_words = self.filter_terms(word_freq, min_freq=5)
        filtered_phrases = self.filter_terms(phrase_freq, min_freq=3)
        
        return {
            'single_words': filtered_words[:100],  # 取前100个
            'noun_phrases': filtered_phrases,  # 取前50个
            'tfidf_terms': tfidf_terms[:50]
        }
    
    def extract_tfidf_terms(self, abstracts: List[str], ngram_range: Tuple[int, int] = (1, 3)) -> List[Tuple[str, float]]:
        """使用TF-IDF提取重要术语"""
        # 自定义tokenizer来保留特定的连字符词
        def custom_tokenizer(text):
            tokens = nltk.word_tokenize(text.lower())
            # 保留包含连字符的术语（如cache-coherence）
            tokens = [token for token in tokens 
                     if token not in self.stop_words and 
                     (len(token) > 2 or '-' in token)]
            return tokens
        
        vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,
            ngram_range=ngram_range,
            max_features=1000,
            stop_words=list(self.stop_words)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算平均TF-IDF分数
            tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # 组合术语和分数
            terms_with_scores = list(zip(feature_names, tfidf_scores))
            # 按分数排序
            terms_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            return terms_with_scores
        except:
            return []
    
    def filter_terms(self, term_counter: Counter, min_freq: int = 3) -> List[Tuple[str, int]]:
        """过滤术语"""
        filtered_terms = []
        
        for term, freq in term_counter.most_common():
            if freq >= min_freq and self.is_valid_term(term):
                filtered_terms.append((term, freq))
        
        return filtered_terms
    
    def is_valid_term(self, term: str) -> bool:
        """检查术语是否有效"""
        # 过滤掉太常见的词
        common_computer_words = {
            'system', 'data', 'model', 'algorithm', 'design', 
            'implementation', 'analysis', 'technique', 'framework'
        }
        
        if term in common_computer_words:
            return False
        
        # 检查是否包含有效字符
        if not re.search(r'[a-z]', term):
            return False
        
        # 过滤掉太短的术语（对于多词短语）
        if len(term.replace('-', '').replace(' ', '')) < 3:
            return False
        
        return True
    
    def post_process_terms(self, extracted_terms: Dict) -> Dict:
        """后处理提取的术语"""
        final_terms = {}
        
        # 合并所有术语，去除重复
        all_terms_set = set()
        
        for category, terms in extracted_terms.items():
            processed_terms = []
            for term, score in terms:
                # 标准化术语格式
                if isinstance(term, tuple):
                    term = term[0] if len(term) > 0 else ""
                
                if term and term not in all_terms_set:
                    all_terms_set.add(term)
                    processed_terms.append((term, score))
            
            final_terms[category] = processed_terms
        
        return final_terms

# 使用示例
def main():
    # 假设 abstracts 是您的摘要列表
    # abstracts = [
    #     "This paper presents a novel cache coherence protocol for many-core processors...",
    #     "We propose a machine learning based prefetcher that significantly improves performance...",
    #     # ... 更多摘要
    # ]

    from loader import load_HPCA, load_ACM
    
    abstracts = load_ACM("ASPLOS") + load_ACM("MICRO") + load_ACM("ISCA") + load_HPCA()
    processed_abstracts = []
    for abstract in abstracts:
        keywords = ["memory", "cache", "dram", "disk", "storage"]
        if any(keyword in abstract.lower() for keyword in keywords):
            processed_abstracts.append(abstract.replace("\n", " "))

    extractor = ComputerTermExtractor()
    important_terms = extractor.extract_technical_terms(processed_abstracts)
    with open("terms_processed.txt", "w") as f:
        for noun_phrase, freq in important_terms["noun_phrases"]:
            f.write(f"{noun_phrase}: {freq}\n")

    processed_terms = extractor.post_process_terms(important_terms)

    0 / 0
    
    # 输出结果
    # print("重要单字词:")
    # for term, freq in processed_terms['single_words'][:20]:
    #     print(f"  {term}: {freq}")
    
    # print("\n重要名词短语:")
    # for term, freq in processed_terms['noun_phrases'][:20]:
    #     print(f"  {term}: {freq}")
    
    print("\nTF-IDF重要术语:")
    for term, score in processed_terms['tfidf_terms'][:20]:
        print(f"  {term}: {score:.4f}")

if __name__ == "__main__":
    main()