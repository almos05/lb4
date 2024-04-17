from lb4_classes import CreateA, CreateB

with open("test", "r", encoding="utf-8") as file:
    text = file.read()

cr_A = CreateA('нет да', text)
cr_A.calculate_tf_and_idf().sort_by_tf_idf().describe_df()
cr_A.information()
print(cr_A.real_text)

cr_B = CreateB('нет да', text)
