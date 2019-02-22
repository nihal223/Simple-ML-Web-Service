import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

path = "./Data/train.csv"
df = pd.read_csv(path)

include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df_ = df[include]

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)


dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]


lr = LogisticRegression()
lr.fit(x,y)


joblib.dump(lr, 'model.pkl')
print('Model Dumped!')

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print('Model Columns Dumped!')
