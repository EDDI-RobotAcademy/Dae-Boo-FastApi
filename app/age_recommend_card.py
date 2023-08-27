import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('card_file.csv')

# 연령대를 원-핫 인코딩
age_one_hot = pd.get_dummies(df['연령대별'], prefix='연령')

# 기존 데이터와 원-핫 인코딩된 데이터 합치기
add_data_one_hot = pd.concat([df, age_one_hot], axis=1)
data = add_data_one_hot.dropna()  # 결측치 제거

# 필요한 컬럼만 선택
age_select_culumns = add_data_one_hot[['연령_10대', '연령_20대', '연령_30대', '연령_40대', '연령_50대',
                               '연령_60대', '연령_70대이상', '카드 번호']]

print(age_select_culumns)

# 훈련 세트와 테스트 세트로 나누기
train_data, test_data = train_test_split(age_select_culumns, random_state=42)

print("훈련 세트 크기:", len(train_data))
print("테스트 세트 크기:", len(test_data))

train_data['카드 번호'] = train_data['카드 번호'].astype(str)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['카드 번호'])
# 텍스트 데이터를 숫자로 표현해서 기계학습 모델에 입력할 수 있게 만듦
vocab_size = len(tokenizer.word_index) + 1
# 모델 구성
max_length = 10  # 텍스트 데이터의 최대 길이
num_age_categories = 7 # 연령대 카테고리 수

card_input = Input(shape=(max_length,)) # 카드 번호 입력받기
embedding_layer = Embedding(vocab_size, 128)(card_input) # 데이터를 밀집 벡터로 변환하는 역할
flatten_layer = Flatten()(embedding_layer) # 일렬로 펼치기

age_input = Input(shape=(num_age_categories,)) # 연령 카테고리 입력받기
merged_layer = Concatenate()([flatten_layer, age_input]) # 카드 번호 데이터와 카테고리 데이터 결합(연결)
dense_layer = Dense(128, activation='relu')(merged_layer) # 128개의 뉴런의 수로 구성된 밀집 연결층에 merged_layer 입력
# ReLu : Rectified Linear Activation 입력값이 0보다 작으면 0으로 반환하고, 0보다 크면 입력값을 그대로 출력하는 함수
# 비선형성을 추가하며, 딥러닝 모델의 복잡한 패턴을 학습할 수 있도록 돕는다.
output_layer = Dense(vocab_size, activation='softmax')(dense_layer) # 다중 클래스 분류를 위해 밀집 연결층 생성
# 카드 번호가 단어 집합의 크기
# dense_layer의 출력을 받아서 vocab_size개의 클래스로 분류

model = Model(inputs=[card_input, age_input], outputs=output_layer)
# card_input, age_input 입력 데이터를 리스트로 나열 후, output_layer 모델 출력
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 모델 컴파일 (학습할 준비 상태)

# 입력 데이터 준비
test_card_data = tokenizer.texts_to_sequences(test_data['카드 번호'].astype(str)) # 전처리
test_card_data = pad_sequences(test_card_data, maxlen=max_length) # 길이 조정


card_train = tokenizer.texts_to_sequences(train_data['카드 번호'])
card_train = pad_sequences(card_train, maxlen=max_length)
age_train = train_data[['연령_10대', '연령_20대', '연령_30대', '연령_40대', '연령_50대', '연령_60대', '연령_70대이상']].values

# 카드 번호를 원-핫 인코딩하여 정답 데이터로 사용
card_numbers = df['카드 번호'].astype(str).tolist() # 카드 번호를 리스트로 저장
label_encoder = LabelEncoder()
label_encoder.fit(card_numbers) # card_numbers 이용해 레이블 인코더 학습
encoded_labels = label_encoder.transform(card_numbers)
one_hot_labels = keras.utils.to_categorical(encoded_labels, num_classes=vocab_size)
# 원 핫 벡터로 변환 // 모델의 입력을 사용할 수 있는 전처리 과정

# 모델 훈련
model.fit([card_train, age_train], one_hot_labels, epochs=10, batch_size=32, validation_split=0.2)



# 모델 예측
test_age_data = test_data[['연령_10대', '연령_20대', '연령_30대', '연령_40대', '연령_50대', '연령_60대', '연령_70대이상']].values
predictions = model.predict([test_card_data, test_age_data])

# 예측 결과 디코딩
predicted_card_indices = np.argmax(predictions, axis=1)
predicted_card_numbers = label_encoder.inverse_transform(predicted_card_indices)

print("예측된 카드 번호:", predicted_card_numbers)