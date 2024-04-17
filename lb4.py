from lb4_classes import CreateA, CreateB

with open("test", "r", encoding="utf-8") as file:
    text = file.read()

cr_A = CreateA('нет да', text)
cr_A.calculate_tf_and_idf().sort_by_tf_idf().describe_df()

cr_B = CreateB('нет да', text)
cr_B.make_tuples()

cr_A.information()
cr_B.information()

res_docs = cr_A.check_df().sort_index()['TF-IDF'] / cr_B.check_df().iloc[:, 0]

for i in res_docs.sort_values(ascending=False).index:
    print(cr_A.real_text[i])
