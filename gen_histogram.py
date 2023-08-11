import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔고딕 폰트 경로 설정
font_path = 'NanumGothic-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 주어진 데이터
bins = ['7.0 ~ 11.5', '11.5 ~ 16.0', '16.0 ~ 20.5', '20.5 ~ 25.0', '25.0 ~ 29.5']
frequencies = [1, 5, 7, 6, 5]

# 히스토그램 그리기
plt.bar(bins, frequencies, color='gray')
plt.xlabel('구간 (Degree ˚)', fontproperties=font_prop)
plt.ylabel('도수', fontproperties=font_prop)
plt.title('히스토그램', fontproperties=font_prop)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.show()