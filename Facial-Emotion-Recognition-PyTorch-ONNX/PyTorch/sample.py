import pandas as pd
df1 = pd.read_csv(r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/train/train.csv")
df2 = pd.read_csv(r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/test/test.csv")

df3 = pd.read_csv(r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/fer2013new.csv")

print(len(df3))
print(df3.head())
df3 = df3.iloc[:, 2:]
print(len(df3))
print(df3.tail())
print((len(df1) + len(df2)) == len(df3))

lis = []

emotion_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
                    4: 'Anger', 5: 'Disguest', 6: 'Fear'}

new_df = pd.DataFrame()
pix = list(df1['pixels'].values) + list(df2['pixels'].values)
new_df['pixels'] = pix
assert (len(df3) == len(new_df))


for i in range(len(df3)):
    vals = df3.iloc[i, :].values
    vals = list(vals)
    max_ind = int(vals.index(max(vals)))

    if(max_ind in [7, 8, 9] or vals.count(vals[max_ind]) > 1):
        lis.append(None)
    else:
        lis.append(max_ind)
print(lis.count(None))
print(len(lis))
for i in range(10):
    print("count  of ", i, " is :", lis.count(i))

new_df['emotions'] = lis
new_df = new_df.dropna(subset=['emotions'])
new_df['emotions'] = new_df['emotions'].astype(int)


print(new_df.tail())
print(len(new_df))
# new_df.to_csv("total_ferplus.csv", index=False)