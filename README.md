# KeyphraseExtraction

- Foder experiment lưu các file log quá trình training và hyperparameter setting cho từng model
- Folder saved_model lưu các checkpoint cho từng model
- Tiền xử lý dữ liệu
```
python KeyPhraseExtraction/src/preprocessing.py [--data_name] [----get_embedding] [--bert_encoding] [--glove_encoding] [--doc_embedding]
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
