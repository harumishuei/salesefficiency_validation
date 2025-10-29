wdu(wdu_single_doc.py)で "data/nissan.plain.txt"を読み込みMilvusに保存し、質問に対してAIが類似文章を検索・回答を生成し、結果をExcelに保存します。
app.pyでは簡易UIから操作可能です。

.envファイルに IBM API キー等を設定
IBM_URL=your_url
IBM_APIKEY=your_apikey
IBM_PROJECT_ID=your_project_id
