import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# CSVファイルのパス
file_path = '20240513.csv'

# CSVファイルを読み込む（shift_jis エンコーディングを使用）
df = pd.read_csv(file_path, encoding='shift_jis')

# 時刻の列を追加
start_time = datetime(2023, 1, 1, 0, 0)  # 開始時刻を設定
df['time'] = [(start_time + timedelta(minutes=i)).strftime('%H:%M') for i in range(len(df))]

# 1分ごとの車両の数をプロット（折れ線グラフ）
plt.figure(figsize=(15, 6))
for column in df.columns[1:-1]:
    plt.plot(df.index, df[column], label=column)
plt.xlabel('Minutes')  # X軸ラベルを設定
plt.ylabel('Number of Vehicles')  # Y軸ラベルを設定
plt.title('Number of Vehicles per Minute')  # グラフタイトルを設定
plt.legend(loc='upper right')  # レジェンドの位置を手動で指定
plt.savefig('vehicles_per_minute_line.png')  # グラフをファイルに保存
plt.show()

# 1分ごとの車両の数をプロット（棒グラフ）
plt.figure(figsize=(15, 6))
for column in df.columns[1:-1]:
    plt.bar(df.index, df[column], label=column, alpha=0.5)  # alphaを使って透明度を設定
plt.xlabel('Minutes')  # X軸ラベルを設定
plt.ylabel('Number of Vehicles')  # Y軸ラベルを設定
plt.title('Number of Vehicles per Minute')  # グラフタイトルを設定
plt.legend(loc='upper right')  # レジェンドの位置を手動で指定
plt.savefig('vehicles_per_minute_bar.png')  # グラフをファイルに保存
plt.show()

# 1時間ごとの車両の数を計算
df['hour'] = df.index // 60  # 分を60で割って時間単位に変換
hourly_sum = df.groupby('hour').sum()  # 時間ごとに合計を計算

# 時刻の列を追加
hourly_sum['time'] = [(start_time + timedelta(hours=i)).strftime('%H:%M') for i in range(len(hourly_sum))]

# 1時間ごとの車両の数をプロット（折れ線グラフ）
plt.figure(figsize=(15, 6))
for column in df.columns[1:-2]:
    plt.plot(hourly_sum['time'], hourly_sum[column], label=column)
plt.xlabel('Time')  # X軸ラベルを設定
plt.ylabel('Number of Vehicles')  # Y軸ラベルを設定
plt.title('Number of Vehicles per Hour')  # グラフタイトルを設定
plt.xticks(rotation=45)  # X軸ラベルを45度回転
plt.legend(loc='upper right')  # レジェンドの位置を手動で指定
plt.savefig('vehicles_per_hour_line.png')  # グラフをファイルに保存
plt.show()

# 1時間ごとの車両の数をプロット（棒グラフ）
plt.figure(figsize=(15, 6))
for column in df.columns[1:-2]:
    plt.bar(hourly_sum['time'], hourly_sum[column], label=column, alpha=0.5)  # alphaを使って透明度を設定
plt.xlabel('Time')  # X軸ラベルを設定
plt.ylabel('Number of Vehicles')  # Y軸ラベルを設定
plt.title('Number of Vehicles per Hour')  # グラフタイトルを設定
plt.xticks(rotation=45)  # X軸ラベルを45度回転
plt.legend(loc='upper right')  # レジェンドの位置を手動で指定
plt.savefig('vehicles_per_hour_bar.png')  # グラフをファイルに保存
plt.show()

# 小複数プロット（1分ごと）
fig, axes = plt.subplots(nrows=len(df.columns[1:-2]), ncols=1, figsize=(15, 3*len(df.columns[1:-2])), sharex=True)
for i, column in enumerate(df.columns[1:-2]):
    axes[i].plot(df.index, df[column], label=column)
    axes[i].set_ylabel('Number of Vehicles')  # Y軸ラベルを設定
    axes[i].legend(loc='upper right')  # レジェンドの位置を手動で指定
axes[-1].set_xlabel('Minutes')  # X軸ラベルを設定
fig.suptitle('Number of Vehicles per Minute (Small Multiple Plots)')  # グラフの総タイトルを設定
plt.savefig('vehicles_per_minute_small_multiples.png')  # グラフをファイルに保存
plt.show()

# 小複数プロット（1時間ごと）
fig, axes = plt.subplots(nrows=len(hourly_sum.columns[1:-1]), ncols=1, figsize=(15, 3*len(hourly_sum.columns[1:-1])), sharex=True)
for i, column in enumerate(hourly_sum.columns[1:-1]):
    axes[i].plot(hourly_sum['time'], hourly_sum[column], label=column)
    axes[i].set_ylabel('Number of Vehicles')  # Y軸ラベルを設定
    axes[i].legend(loc='upper right')  # レジェンドの位置を手動で指定
axes[-1].set_xlabel('Time')  # X軸ラベルを設定
fig.suptitle('Number of Vehicles per Hour (Small Multiple Plots)')  # グラフの総タイトルを設定
plt.xticks(rotation=45)  # X軸ラベルを45度回転
plt.savefig('vehicles_per_hour_small_multiples.png')  # グラフをファイルに保存
plt.show()
