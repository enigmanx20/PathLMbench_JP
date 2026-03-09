import re
import random
from collections import Counter, defaultdict
import csv

import MeCab
import jaconv

def ngram(text, n=3):
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def ngram_overlap(ref, pred, n=3, return_counts=False):
    ref_ngrams = Counter(ngram(ref, n))
    pred_ngrams = Counter(ngram(pred, n))

    intersection = sum((ref_ngrams & pred_ngrams).values())
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = intersection / total_pred if total_pred else 0
    recall = intersection / total_ref if total_ref else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    # multiset jaccard
    union = sum((ref_ngrams | pred_ngrams).values())
    jaccard = intersection / union if union else 0

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }

    if return_counts:
        result.update({
            "intersection": intersection,
            "total_pred": total_pred,
            "total_ref": total_ref,
            "union": union
        })

    return result


def tokenize_with_mecab(text: str, tagger: MeCab.Tagger) -> list:
    """MeCabで形態素解析し、表層形のリストを返す"""
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface:
            tokens.append(node.surface)
        node = node.next
    return tokens


def ngram_from_tokens(tokens: list, n: int = 2) -> list:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def evaluate_ngram_overlap_mecab(ref_text: str, pred_text: str, tagger: MeCab.Tagger, n: int = 2, return_counts=False):
    ref_tokens = tokenize_with_mecab(ref_text, tagger)
    pred_tokens = tokenize_with_mecab(pred_text, tagger)

    ref_ngrams = Counter(ngram_from_tokens(ref_tokens, n))
    pred_ngrams = Counter(ngram_from_tokens(pred_tokens, n))

    intersection = sum((ref_ngrams & pred_ngrams).values())
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = intersection / total_pred if total_pred else 0
    recall = intersection / total_ref if total_ref else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    # multiset jaccard
    union = sum((ref_ngrams | pred_ngrams).values())
    jaccard = intersection / union if union else 0

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }

    if return_counts:
        result.update({
            "intersection": intersection,
            "total_pred": total_pred,
            "total_ref": total_ref,
            "union": union
        })

    return result


def compute_micro_average(results_with_counts):
    total_intersection = 0
    total_pred = 0
    total_ref = 0
    total_union = 0

    for res in results_with_counts:
        total_intersection += res.get("intersection", 0)
        total_pred += res.get("total_pred", 0)
        total_ref += res.get("total_ref", 0)
        total_union += res.get("union", 0)

    precision = total_intersection / total_pred if total_pred else 0
    recall = total_intersection / total_ref if total_ref else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    jaccard = total_intersection / total_union if total_union else 0

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "micro_jaccard": jaccard
    }

def compute_metrics(results_mecab_with_counts, results_char_with_counts):
    # ----------------------
    # Macro & Micro: MeCab
    # ----------------------
    macro_precision_m = sum(d["precision"] for d in results_mecab_with_counts) / len(results_mecab_with_counts)
    macro_recall_m    = sum(d["recall"]    for d in results_mecab_with_counts) / len(results_mecab_with_counts)
    macro_f1_m        = sum(d["f1"]        for d in results_mecab_with_counts) / len(results_mecab_with_counts)
    macro_jaccard_m   = sum(d["jaccard"]   for d in results_mecab_with_counts) / len(results_mecab_with_counts)
    
    micro_scores_m = compute_micro_average(results_mecab_with_counts)
    
    # ----------------------
    # Macro & Micro: Char
    # ----------------------
    macro_precision_c = sum(d["precision"] for d in results_char_with_counts) / len(results_char_with_counts)
    macro_recall_c    = sum(d["recall"]    for d in results_char_with_counts) / len(results_char_with_counts)
    macro_f1_c        = sum(d["f1"]        for d in results_char_with_counts) / len(results_char_with_counts)
    macro_jaccard_c   = sum(d["jaccard"]   for d in results_char_with_counts) / len(results_char_with_counts)
    
    micro_scores_c = compute_micro_average(results_char_with_counts)
    
    # ----------------------
    # 出力
    # ----------------------
    print(f"Total dataset length: {len(results_mecab_with_counts)}")
    print("===== MeCab n-gram =====")
    print(f"[Macro] Precision: {macro_precision_m:.3f}, Recall: {macro_recall_m:.3f}, F1: {macro_f1_m:.3f}, Jaccard: {macro_jaccard_m:.3f}")
    print(f"[Micro] Precision: {micro_scores_m['micro_precision']:.3f}, Recall: {micro_scores_m['micro_recall']:.3f}, F1: {micro_scores_m['micro_f1']:.3f}, Jaccard: {micro_scores_m['micro_jaccard']:.3f}")
    
    print("\n===== Char n-gram =====")
    print(f"[Macro] Precision: {macro_precision_c:.3f}, Recall: {macro_recall_c:.3f}, F1: {macro_f1_c:.3f}, Jaccard: {macro_jaccard_c:.3f}")
    print(f"[Micro] Precision: {micro_scores_c['micro_precision']:.3f}, Recall: {micro_scores_c['micro_recall']:.3f}, F1: {micro_scores_c['micro_f1']:.3f}, Jaccard: {micro_scores_c['micro_jaccard']:.3f}")
    

ignore_pattern = re.compile(
    r'^('
    r'mm|cm'                                                        # 単位（文字列）← 先に置く
    r'|㎜|㎝|㎞|㎎|㎏|㌘|㍉|㌔|㍍'                                   # 単位（記号）
    r'|≒|№'                                                           # 特殊記号
    r'|[。、.,:()\[\]%#/\-【】＞＜～・=×ｘ+「」『』〔〕［］〈〉《》{}'  # 括弧や記号類（その1）
    r'Ⅰ-Ⅻ①-⑳→←↑↓><“”’○●＊※±]'                                    # 括弧や記号類（その2：記号と記号文字）
    r'|[０-９0-9]+([．.][０-９0-9]+)?'                                # 数字（小数含む）
    r'|[A-Za-z]+'                                                    # 英単語全般
    r')$'
)
#===========================================================================================================================================
def _format_text(concat):
    jp_char = r'[一-龯ぁ-んァ-ン]'

    concat = re.sub(r'(?<=\d)x(?=\d)', '×', concat)
    concat = re.sub(r'＋', '+', concat)
    #concat = re.sub(r'ー', '-', concat)
    concat = re.sub(r'([+*/=()])', r' \1 ', concat)
    concat = re.sub(r'[\n\t]', ',', concat)
    concat = re.sub(r'\u3000', ' ', concat)
    concat = re.sub(r'[ \u3000]+', ' ', concat)
    concat = re.sub('（', '(', concat)
    concat = re.sub('）', ')', concat)
    concat = re.sub(r'\),', ')', concat)
    concat = re.sub('】,', ')', concat)
    concat = re.sub('：', ':', concat)
    concat = re.sub('；', ':', concat)
    concat = re.sub(';', ':', concat)
    concat = re.sub(',{2,}', ',', concat)
    concat = re.sub(', ', ',', concat)
    #concat = re.sub('、', ',', concat)
    concat = re.sub('＃', '#', concat)
    concat = re.sub(r'。[,、。]+', '。', concat)
    concat = re.sub(r'[,、。]+。', '。', concat)
    # 左側が日本語：日本語 + , → 日本語 + 、
    concat = re.sub(f'({jp_char}),', r'\1、', concat)
    # 右側が日本語：, + 日本語 → 、 + 日本語
    concat = re.sub(f',({jp_char})', r'、\1', concat)
    # 左側が日本語：日本語 + , → 日本語 + 、
    concat = re.sub(f'({jp_char}):', r'\1：', concat)
    # 右側が日本語：, + 日本語 → 、 + 日本語
    concat = re.sub(f':({jp_char})', r'：\1', concat)
    concat = re.sub(r':,', ':', concat)
    concat = re.sub(r'彎', '弯', concat)
    concat = re.sub('％', '%', concat)
    concat = re.sub(r'№', 'No.', concat)
    concat = concat.translate(str.maketrans({
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
    'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
    'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
    'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
    'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
    'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
    'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
    'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
    'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
    'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z',
    '－': '-', '！': '!', '？': '?', '＠': '@', '＊': '*',
    '＆': '&', '＿': '_', '＋': '+', '＝': '=', '＜': '<',
    '＞': '>', '［': '[', '］': ']', '｛': '{', '｝': '}',
    '＼': '\\', '｜': '|', '＾': '^', '＄': '$', '＃': '#',
    '”': '"', '’': "'", '｀': '`'
}))
    concat = jaconv.h2z(concat, kana=True)
    concat = re.sub(r' ', ' SPACE ', concat)
    return concat

def format_text(concat):
    jp_char = r'[一-龯ぁ-んァ-ン]'
    concat = re.sub(r'(?<=\d)x(?=\d)', '×', concat)
    concat = re.sub(r'＋', '+', concat)
    #concat = re.sub(r'ー', '-', concat)
    
    # (3) 修正：括弧 () には空白を入れない（+*/= のみに限定）
    #concat = re.sub(r'([+*/=])', r' \1 ', concat)
    
    concat = re.sub(r'[\n\t]', ',', concat)
    concat = re.sub(r'\u3000', ' ', concat)
    concat = re.sub(r'[ \u3000]+', ' ', concat)
    concat = re.sub('（', '(', concat)
    concat = re.sub('）', ')', concat)
    concat = re.sub(r'\),', ')', concat)
    
    # (4) 修正：】, を ) にしない（カンマだけ除去して 】 を保持）
    concat = re.sub('】,', '】', concat)
    
    concat = re.sub('：', ':', concat)
    concat = re.sub('；', ':', concat)
    concat = re.sub(';', ':', concat)
    concat = re.sub(',{2,}', ',', concat)
    concat = re.sub(', ', ',', concat)
    #concat = re.sub('、', ',', concat)
    concat = re.sub('＃', '#', concat)
    concat = re.sub(r'。[,、。]+', '。', concat)
    concat = re.sub(r'[,、。]+。', '。', concat)
    # 左側が日本語：日本語 + , → 日本語 + 、
    concat = re.sub(f'({jp_char}),', r'\1、', concat)
    # 右側が日本語：, + 日本語 → 、 + 日本語
    concat = re.sub(f',({jp_char})', r'、\1', concat)
    # 左側が日本語：日本語 + : → 日本語 + ：
    concat = re.sub(f'({jp_char}):', r'\1：', concat)
    # 右側が日本語：: + 日本語 → ： + 日本語
    concat = re.sub(f':({jp_char})', r'：\1', concat)
    concat = re.sub(r':,', ':', concat)
    concat = re.sub(r'彎', '弯', concat)
    concat = re.sub('％', '%', concat)
    concat = re.sub(r'№', 'No.', concat)
    concat = concat.translate(str.maketrans({
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
        'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
        'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
        'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
        'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
        'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
        'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
        'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
        'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z',
        '－': '-', '！': '!', '？': '?', '＠': '@', '＊': '*',
        '＆': '&', '＿': '_', '＋': '+', '＝': '=', '＜': '<',
        '＞': '>', '［': '[', '］': ']', '｛': '{', '｝': '}',
        '＼': '\\', '｜': '|', '＾': '^', '＄': '$', '＃': '#',
        '”': '"', '’': "'", '｀': '`'
    }))
    concat = jaconv.h2z(concat, kana=True)
    concat = re.sub(r' ', ' SPACE ', concat)
    return concat
    

def should_ignore(token, ignore_pattern):
    return bool(ignore_pattern.fullmatch(token))

# --- MeCab による未知語・誤字候補の検出 ---
def detect_mecab_normalies_anomalies(text, dic_dir, usr_dir, ignore=re.compile(r'')):
    tagger = MeCab.Tagger(f'-r /dev/null -d "{dic_dir}" -u "{usr_dir}"')
    tagger.parse('')
    node = tagger.parseToNode(text)

    results_nor = []
    results_ano = []
    
    while node:
        surface = node.surface
        feature = node.feature.split(",")
        pos = feature[0]
        base_form = feature[6] if len(feature) > 6 else '*'
        
        if base_form == '*' or pos == '記号':
            if not should_ignore(surface, ignore):
                results_ano.append(surface)
        else:
            results_nor.append(surface)
        
        node = node.next
    return {'normalies': results_nor, 'anormalies': results_ano}

def normalize_whitespace(text):
    text = text.replace("\u3000", " ")   # 全角スペース
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()

def normalize_symbols(text):
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*x\s*", "x", text)
    return text

import unicodedata

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def normalize_unicode_symbol_white(text, white=True, symbol=True, unicode=True):
    if white:
        text = normalize_unicode(text)
    if symbol:
        text = normalize_symbols(text)
    if unicode:
        text = normalize_whitespace(text)
    return text

#===========================================================================================================================================

def is_english(text):
    """テキストが英語で構成されているか判定"""
    return re.fullmatch(r"[A-Za-z\s.,!?']+", text) is not None

def introduce_typo(word):
    """単語に1つのランダムなタイポを加え、どのような変化があったかも返す"""
    typo_chars = 'abcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-=[]{}|;:,.<>?/'
    if len(word) <= 2:
        return word, []

    typo_type = random.choice(['swap', 'delete', 'insert', 'replace'])
    idx = random.randint(0, len(word) - 2)

    if typo_type == 'swap':
        new_word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    elif typo_type == 'delete':
        new_word = word[:idx] + word[idx+1:]
    elif typo_type == 'insert':
        char = random.choice(typo_chars)
        new_word = word[:idx] + char + word[idx:]
    elif typo_type == 'replace':
        char = random.choice(typo_chars)
        new_word = word[:idx] + char + word[idx+1:]
    else:
        new_word = word
    
    return new_word, [{
        "type": typo_type,
        "position": idx,
        "original": word,
        "modified": new_word
    }]

def add_typos_to_english_word(word, typo_rate=0.1):
    """
    単語が英語であれば、ランダムにタイプミスを加える。
    タイポの最大数は単語長の半分まで。すべてのタイポの詳細を記録する。
    """
    typo_word = word
    typo_log = []

    for _ in range(max(1, len(word) // 2)):
        if random.random() < typo_rate:
            typo_word, typo_info = introduce_typo(typo_word)
            typo_log += typo_info

    return typo_word, typo_log

#===========================================================================================================================================
def is_japanese(text):
    """
    文字列に日本語（ひらがな・カタカナ・漢字）が含まれているか判定。
    """
    return re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text) is not None

# すべてのIPADIC CSVを読み込んで同音異義語辞書を作る
def build_homophones_dict(ipadic_files):
    homophones = defaultdict(set)

    for fname in ipadic_files:
        if fname.endswith(".csv"):
            with open(fname, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 13:
                        continue
                    surface = row[0]        # 表層形（漢字等）
                    reading = row[11]      # 読み（カタカナ）
                    if surface and reading:
                        homophones[reading].add(surface)

    return {k: list(v) for k, v in homophones.items() if len(v) > 1}

def build_word_to_readings_dict(ipadic_files):
    """
    IPADIC形式（UTF-8）のCSV群から、単語→読みの辞書を構築する。
    """
    word_to_readings = defaultdict(set)

    for fname in ipadic_files:
        if fname.endswith(".csv"):
            with open(fname, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 12:
                        continue
                    surface = row[0]
                    reading = row[11]
                    if surface and reading:
                        word_to_readings[surface].add(reading)

    # set → list に変換して返す
    return {k: list(v) for k, v in word_to_readings.items()}

# 誤変換を挿入する関数
def insert_kanji_isconversion(word, reading, homophones):
    """
    同音異義語による誤変換。homophonesに候補があれば誤変換。
    """
    if reading in homophones and word in homophones[reading]:
        candidates = [w for w in homophones[reading] if w != word]
        #candidates = [w for w in homophones[reading]]
        if candidates:
            new_word = random.choice(candidates)
            return new_word, {
                "type": "kanji_misconversion",
                "original": word,
                "modified": new_word,
                "reading": reading,
            }

    return word, {
        "type": "none",
        "original": word,
        "modified": word,
        "reading": reading,
    }
#---------------------precompute homophones from ipa dict and word_dict-------------------------
ipadic_files = ["./dictionaries/ipadic_070610/merged_utf8.csv",
                "./dictionaries/word_dict.csv"]
homophones = build_homophones_dict(ipadic_files)
word_to_readings = build_word_to_readings_dict(ipadic_files)
#-----------------------------------------------------------------------------------------------

def edit_typo(word):
    kana_chars = (
    "ぁあぃいぅうぇえぉお"
    "かがきぎくぐけげこご"
    "さざしじすずせぜそぞ"
    "ただちぢっつづてでとど"
    "なにぬねの"
    "はばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめも"
    "やゃゆゅよょ"
    "らりるれろ"
    "わをんー"
    "ァアィイゥウェエォオ"
    "カガキギクグケゲコゴ"
    "サザシジスズセゼソゾ"
    "タダチヂッツヅテデトド"
    "ナニヌネノ"
    "ハバパヒビピフブプヘベペホボポ"
    "マミムメモ"
    "ヤャユュヨョ"
    "ラリルレロ"
    "ワヲンー"
)
    if len(word) <= 1:
        return word, []
    typo_type = random.choice(['swap', 'delete', 'insert', 'kanji_misconversion'])
    idx = random.randint(0, len(word) - 2)

    if typo_type == 'swap':
        typo_word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    elif typo_type == 'delete':
        typo_word = word[:idx] + word[idx+1:]
    elif typo_type == 'insert':
        char = random.choice(kana_chars)
        typo_word = word[:idx] + char + word[idx:]
    elif typo_type == 'kanji_misconversion':
        try:
            typo_word = insert_kanji_isconversion(word, word_to_readings[word][0], homophones)[1]['modified']
        except:
            print(f"Exception {word}")
            typo_word = word
    else:
        typo_word = word

    return typo_word, [{"type": typo_type, "position": idx, 'original': word, "modified": typo_word}]


# 複合的な処理（一定確率で置換 or 編集）
def add_typos_to_japanese_word(word, typo_rate=0.1):
    """
    単語が日本語であれば、ランダムにタイプミスを加える。
    タイポの最大数は単語長の半分まで。すべてのタイポの詳細を記録する。
    """
    typo_word = word
    typo_log = []

    for _ in range(max(1, len(word) // 2)):
        if random.random() < typo_rate:
            typo_word, log = edit_typo(typo_word)
            typo_log += log

    return typo_word, typo_log

def add_typos_to_text(text, typo_rate, dic_dir, usr_dir, ignore=re.compile(r''), ):
    tagger = MeCab.Tagger(f'-r /dev/null -d "{dic_dir}" -u "{usr_dir}"')
    tagger.parse('')
    node = tagger.parseToNode(text)

    typo_words = []
    typo_logs = []
    
    while node:
        surface = node.surface
        feature = node.feature.split(",")
        pos = feature[0]
        base_form = feature[6] if len(feature) > 6 else '*'

        if base_form == '*' or pos == '記号' or should_ignore(surface, ignore):
            typo_words.append(surface)
        else:
            if is_english(surface):
                surface, typo_log = add_typos_to_english_word(surface, typo_rate)
                typo_logs += typo_log
            elif is_japanese(surface):
                surface, typo_log = add_typos_to_japanese_word(surface, typo_rate)
                typo_logs += typo_log
            else:
                typo_logs += []
            typo_words.append(surface)
        
        node = node.next
    return typo_words, typo_logs

def make_typo(concat, typo_rate, dic_dir, usr_dir):
    concat = format_text(concat)
    concat = concat.strip()
    results = add_typos_to_text(concat, typo_rate, dic_dir, usr_dir, ignore_pattern)
    typo_text = results[0]
    typo_logs = results[1]
    
    #typo_text_concat = "".join(token if token != "SPACE" else " " for token in typo_text)
    typo_text_concat = "".join(typo_text)
    typo_text_concat = re.sub('SPACE', ' ', typo_text_concat)
    return typo_text, typo_text_concat, typo_logs

#=============================================================================================-
import difflib
from typing import List, Dict, Tuple, Any

def _merge_adjacent_ops(ops: List[Tuple[str,int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    """
    Merge consecutive non-equal opcodes to stabilize diff granularity.
    Returns merged blocks as (i1,i2,j1,j2) on (typo_text, corrected_text).
    """
    merged = []
    cur = None  # (i1,i2,j1,j2)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            if cur is not None:
                merged.append(cur)
                cur = None
            continue
        if cur is None:
            cur = (i1, i2, j1, j2)
        else:
            ci1, ci2, cj1, cj2 = cur
            # Extend to cover this adjacent/overlapping change block
            cur = (ci1, max(ci2, i2), cj1, max(cj2, j2))
    if cur is not None:
        merged.append(cur)
    return merged

def pred_changes_from_llm_json(typo_text: str, llm_json: Dict[str, Any], merge_adjacent: bool = True) -> List[Dict[str, Any]]:
    """
    Build position-aware predicted changes from typo_text and LLM JSON that contains:
      {"corrected_text": "...", "changes":[{"before":"...","after":"..."}]}
    We ignore llm_json["changes"] for scoring and derive changes via diff:
      position: start index in typo_text
      modified: substring in typo_text
      original: corresponding substring in corrected_text
    """
    corrected_text = llm_json.get("corrected_text", "")
    sm = difflib.SequenceMatcher(a=typo_text, b=corrected_text)
    opcodes = sm.get_opcodes()

    blocks = _merge_adjacent_ops(opcodes) if merge_adjacent else [
        (i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag != "equal"
    ]

    preds = []
    for i1, i2, j1, j2 in blocks:
        modified = typo_text[i1:i2]
        original = corrected_text[j1:j2]
        preds.append({"position": i1, "modified": modified, "original": original})
    return preds

def _key(change: Dict[str, Any]) -> Tuple[int, str, str]:
    return (int(change["position"]), change.get("modified", ""), change.get("original", ""))

def score_changes(gold_changes: List[Dict[str, Any]], pred_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Scores by exact match on (position, modified, original).
    Returns tp/fp/fn and precision/recall/f1.
    """
    gset = set(_key(c) for c in gold_changes)
    pset = set(_key(c) for c in pred_changes)

    tp_set = gset & pset
    fp_set = pset - gset
    fn_set = gset - pset

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "tp_items": sorted(tp_set),
        "fp_items": sorted(fp_set),
        "fn_items": sorted(fn_set),
    }

def benchmark_one(typo_text: str, clean_text: str, gold_changes: List[Dict[str, Any]], llm_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper:
      - exact match of final text
      - predicted changes from diff
      - change-level precision/recall/f1
    """
    corrected_text = llm_json.get("corrected_text", "")
    pred_changes = pred_changes_from_llm_json(typo_text, llm_json, merge_adjacent=True)
    change_scores = score_changes(gold_changes, pred_changes)

    return {
        "exact_text_match": (corrected_text == clean_text),
        "gold_count": len(gold_changes),
        "pred_count": len(pred_changes),
        **change_scores,
    }

def show_debug(typo_text, clean_text, gold_changes, llm_json):
    pred_changes_merge = pred_changes_from_llm_json(typo_text, llm_json, merge_adjacent=True)
    pred_changes_raw   = pred_changes_from_llm_json(typo_text, llm_json, merge_adjacent=False)

    print("== sanity ==")
    print("typo_text == clean_text ?", typo_text == clean_text)
    print("llm corrected == clean_text ?", llm_json.get("corrected_text","") == clean_text)
    print()

    def clip(s, a, b):
        a = max(0, a); b = min(len(s), b)
        return s[a:b]

    print("== GOLD (with context) ==")
    for g in gold_changes:
        pos = int(g["position"])
        mod = g.get("modified","")
        org = g.get("original","")
        ctx = clip(typo_text, pos-10, pos+len(mod)+10)
        actual = typo_text[pos:pos+len(mod)]
        print({"position": pos, "modified": mod, "original": org, "actual_at_pos": actual, "ctx": ctx})
    print()

    print("== PRED (merge_adjacent=True) ==")
    for p in pred_changes_merge:
        pos = int(p["position"])
        mod = p["modified"]
        org = p["original"]
        ctx = clip(typo_text, pos-10, pos+len(mod)+10)
        print({"position": pos, "modified": mod, "original": org, "ctx": ctx})
    print()

    print("== PRED (merge_adjacent=False) ==")
    for p in pred_changes_raw:
        pos = int(p["position"])
        mod = p["modified"]
        org = p["original"]
        ctx = clip(typo_text, pos-10, pos+len(mod)+10)
        print({"position": pos, "modified": mod, "original": org, "ctx": ctx})
    print()

    print("== SCORE (merge) ==")
    print(score_changes(gold_changes, pred_changes_merge))
    print("== SCORE (raw) ==")
    print(score_changes(gold_changes, pred_changes_raw))
    return None