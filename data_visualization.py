import pandas as pd


ST01_Commercial_NRT_DF = pd.read_excel("Commercial_NRT Technical Description.xlsx", engine='openpyxl')

ST_01_Commercial_NRT_IAR_FY23_DF = pd.read_excel("/Users/louisciccone/microsoft/Data-Description-Quality/ST-01_Commercial_NRT_IAR_FY23.xlsx", engine='openpyxl')


print(ST01_Commercial_NRT_DF.head)

print("next:")
print()
print(ST_01_Commercial_NRT_IAR_FY23_DF.head)

