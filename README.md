# KeyphraseExtraction
- Tải data theo nguồn: https://github.com/LIAAD/KeywordExtractor-Datasets và lưu tại data folder
- Foder experiment lưu các file log quá trình training và hyperparameter setting cho từng model
- Folder saved_model lưu các checkpoint cho từng model: https://drive.google.com/drive/folders/1Gj8gWk-4XeyPOaaAIdQakJczveYwR0fa?fbclid=IwAR3L4Gb9t4jz2pW-qDwLEEPOvsjrdtMluMAcIWDHGoC_xjkxvVgEsW9QEew
- Tiền xử lý dữ liệu
```
python KeyPhraseExtraction/src/preprocessing.py [--data_name] [--get_embedding] [--bert_encoding] [--glove_encoding] [--doc_embedding]
```

- Huấn luyện mô hình:
```
python KeyPhraseExtraction/src/train_{model_name}.py [--args value]
```

- Chạy app:
```
cd KeyPhraseExtraction
streamlit run src/app.py
```
