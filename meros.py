# 走れメロスの文章中からエンティティを抽出する
# https://www.aozora.gr.jp/cards/000035/files/1567_14913.html : からメロスのテキストは取得できるよ

import pandas as pd
import spacy
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("ja_ginza")    # Language クラス


# ------テキストファイルを読み込む ------
def open_file(src_file):
    with open(src_file, "r", encoding = "Shift_JIS") as file:
        text = file.read()
    text = re.sub(r"-+\s.*?\s-+", "", text, flags = re.S)   # +: 1回以上，\s: 空白や改行，.*?: 最短一致，flags = re.S: "."が改行にもマッチするようにするフラグ
    text = re.sub(r"底本.*", "", text, flags = re.S)   # +: 1回以上，\s: 空白や改行，.*?: 最短一致，flags = re.S: "."が改行にもマッチするようにするフラグ
    # print(text)     # 本文を表示
    sentences = [sentence + "。" for sentence in text.split("。") if sentence.strip()]  # "。"で1文ごとに区切る
    # sentences = [sentence.replace("。", "") for sentence in sentences]
    return sentences



if __name__ == "__main__":
    text = open_file("./src/hashire_merosu.txt")    # テキストを取得

    # ------データフレームへと変換 ------
    df = pd.DataFrame({
        "sentence_id": range(len(text)),
        "sentence": text
    })
    df = (df
            .assign(sentence = lambda df: df["sentence"].str.split("。|\n|、"))[["sentence_id", "sentence"]]
            .explode("sentence")    # 各要素を新しい行として展開する
            .assign(
                sentence = lambda df: 
                    df["sentence"]
                        .str.normalize("NFKC")  # ユニコード正規化
                        .str.replace(r"《.*?》|\[#.*?\]|[＃「」()]", "", regex = True)  # 正規表現による不要な表現の削除
                        .str.strip()    # 先頭や末尾のスペース削除
            )
            .query("sentence.str.len()> = 1")   # 要素が空白の行を削除
        )
    # print(df)   # debug 用


    # ----- 形態素解析 -----
    df_token = (
        df.assign(token = lambda df: list(nlp.pipe(df["sentence"])))
        .explode("token")
        .assign(
            text = lambda df: df["token"].apply(lambda e: e.text),  # 元の文章での表記
            lemma = lambda df: df["token"].apply(lambda e: e.lemma_),   # 単語の原型    
            pos = lambda df: df["token"].apply(lambda e: e.pos_),  # 単語の品詞
            is_stop = lambda df: df["token"].apply(lambda e: e.is_stop),     # 単語がストップワードかを表すブール値
            has_vector = lambda df: df["token"].apply(lambda e: e.has_vector),  # 単語にベクトル表現が付与されているかどうかを表すブール値
            vector = lambda df: df["token"].apply(lambda e: e.vector)   # 単語のベクトル表現(なければ0ベクトル)
        )
        .drop(columns = "token")    # token 列の削除
        .reset_index(drop = True)   # index を修正する
    )
    # print(df_token.columns)         # ['sentence_id', 'sentence', 'text', 'lemma', 'pos', 'is_stop','has_vector', 'vector'],
    # print(df_token)
    # df_token.to_csv("./meros_dataFrame.csv", index=True)


    # ----- 単語間の類似度を比較する -----
    # 名詞と固有名詞を取り出す
    df_token = df_token.query("pos in ('NOUN', 'PROPN') and not is_stop")
    print(df_token)
    # df_token.to_csv("./noun.csv", index=True)

    # 重複していないエンティティのリストを作成する
    df_token = df_token.query("has_vector == True")     # 単語ベクトルが存在しない単語を削除
    unique_tokens = df_token["text"].unique()     # 重複したtokenを削除する
    vectors = np.array([
        df_token[df_token["text"] == token].iloc[0]["vector"] for token in unique_tokens     # tokenに対応するベクトルをget
    ])

    similarity_matrix = cosine_similarity(vectors)  # cosine類似度を計算
    df_similarity = pd.DataFrame(similarity_matrix, index = unique_tokens, columns = unique_tokens)
    # df_similarity.to_csv("./similarity.csv", index=True)
    print(df_similarity)
    print(df_similarity["おまえ"].sort_values(ascending=False).head(10))     # メロスとの類似度Top5を表示
    # print(df_token)    

