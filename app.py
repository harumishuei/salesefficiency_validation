import gradio as gr
from extract import load_and_chunk_file, get_embeddings, setup_milvus, custom_llm

def run_pipeline(file, query):
    # ファイル読み込みとチャンク化
    texts = load_and_chunk_file(file.name)
    vectors = get_embeddings(texts)
    milvus = setup_milvus(vectors, texts)

    # クエリベクトル化と検索
    query_vector = get_embeddings([query])[0]
    search_results = milvus.search(
        collection_name="yuhou_collection",
        data=[query_vector],
        limit=5,
        output_fields=["id", "text"]
    )
    hits = search_results[0]

    # 根拠整形
    context = "\n\n".join([
        f"根拠 {i+1}（ID: {hit['id']}）\n" + hit["text"].strip()
        for i, hit in enumerate(hits)
    ])

    # プロンプト生成
    prompt = f"""
    ## System
    あなたは財務分析に精通したアシスタントです。
    ユーザーの指示に可能な限り正確に従ってください。
    知ったかぶりをせず、事実に基づいて回答してください。
    必ず日本語で回答してください。

    ## User
    以下の条件に従ってください。

    1. 質問に対する回答がわからない場合は「わかりません」と答えてください。誤った情報は絶対に共有しないでください。
    2. 以下の根拠には、企業の財務指標や経営データが記載されています。
    3. 質問文からキーワードを抽出してください（例：「未払金」「買掛金」など）。
    4. キーワードの同義語・類義語も抽出してください。
    - 「退職給付費用」および「退職一時金」は「退職金」と同義として扱うこと。
    5. 根拠の中から、抽出したキーワードおよび同義語を含むすべての記述（表の該当行や文章）を抜き出してください。
    6. 関連する根拠（同じ勘定科目や関連財務項目など）も含めて抽出してください。
    7. 抽出した根拠に基づき、該当する数値や説明を簡潔にまとめてください。
    8. 数値の計算が必要な場合は、根拠に記載された算出方法に従って計算し、**計算のステップ（式や途中結果）と最終的な計算結果**を示してください。
    9. 回答は最後まで出力してください。

    質問:
    {query}

    根拠:
    {context}

    回答:
    """
    response = custom_llm.invoke(prompt)

    return context, response

# Gradio UI定義
demo = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.File(label="財務データファイル（.txt or .json）"),
        gr.Textbox(label="質問を入力", placeholder="例：退職金はいくらですか？")
    ],
    outputs=[
        gr.Textbox(label="根拠（検索結果）", lines=10),
        gr.Textbox(label="回答（AI生成）", lines=10)
    ],
    title="財務分析AIアシスタント",
    description="有価証券報告書をもとに、質問に対して根拠付きで回答します。"
)

demo.launch()
