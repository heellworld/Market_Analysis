import re
import string
from underthesea import word_tokenize  # Thay đổi này phụ thuộc vào thư viện bạn sử dụng
import emoji  # Đảm bảo rằng thư viện này được cài đặt

lookup_dict = {'sp':'sản phẩm', 'dt':'điện thoại', "ko" : "không", "seal" : "tem_mới", "gút chóp" : "tốt", "m" : "mình", "ok" : "tốt", "chẹp" : "đẹp",
               "đc" : "được","dc" : "được", "chất" : "tốt","gudddddddddddddddddddd":"tốt", "vs" : "với", "ng" : "người", "triệu like":"quá tốt", "feedback":"phản hồi",
               "ngon lành cành đào":"quá tốt", "dag":"đang","qc":"quảng cáo","k" : "không","hd sd": "hướng dẫn sử dụng","sd" : "sử dụng","hd": "hướng dẫn",
               "nt":"nhắn tin"}

def convert_emojis(text):
    if isinstance(text, str):
        return emoji.demojize(text, delimiters=("", ""))
    else:
        return text

def clean_text(text, lookup_dict):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = convert_emojis(text)
        # Loại bỏ các dấu câu
        text = re.sub(r'[\{\}\[\]\(\)\.,!@#\$%\^&\*?><:";\'\|\\=\+\-_]', ' ', text)
        # Loại bỏ ký tự lặp lại
        text = re.sub(r'(.)\1+', r'\1', text)
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        normalized_words = [lookup_dict.get(word, word) for word in words]
        normalized_text = ' '.join(normalized_words)
        # Sử dụng underthesea để tách từ tiếng Việt
        tokenized_text = word_tokenize(normalized_text, format="text")
        return tokenized_text
    else:
        return text
