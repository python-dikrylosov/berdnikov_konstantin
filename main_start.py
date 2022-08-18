# -*- coding: utf-8 -*-
import math
import os
os.system("Color 0a")
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
import time
start_time = time.time()
# BTC/AUD,
# BTC/BIDR,
# BTC/BRL,
# BTC/BUSD,
# BTC/EUR,
# BTC/GBP,
# BTC/RUB,
# BTC/TRY,
# BTC/TUSD,
# BTC/UAH,
# BTC/USDC,
# BTC/USDP,
# BTC/USDT,
# https://python-binance.readthedocs.io/en/latest/overview.html
import password
from binance.client import Client
client = Client(password.api_key, password.secret_key)
min_lovume_usdt = input("Введите сумму ордера в $ мин 11 : ")
import yfinance as yf
import random
import pandas as pd
import numpy as np
from PIL import ImageGrab
import matplotlib.pyplot as plt
import cv2

while True:
    info = client.get_all_tickers()
    real_date = time.strftime("%Y-%m-%d")
    time_minute: str = time.strftime("%M")
    print([time.strftime("%Y-%m-%d %H:%M:%S"),time_minute])

    def neurostart():
        info = client.get_all_tickers()

        # data_BTCUSDT_yf = yf.download("BTC-USD", start="2022-08-01", end=real_date, interval="1m")
        all_BTCUSDT = info[11]
        price_BTCUSDT = all_BTCUSDT["price"]
        data_BTCUSDT_safe_real_kurs = open("BTC-USD_2022-08-01_1m.csv", "a")
        data_BTCUSDT_safe_real_kurs.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
        data_BTCUSDT_safe_real_kurs.write(",")
        data_BTCUSDT_safe_real_kurs.write(str(price_BTCUSDT))
        data_BTCUSDT_safe_real_kurs.write(",")
        data_BTCUSDT_safe_real_kurs.write("\n")
        data_BTCUSDT_safe_real_kurs.close()

        data_BTCUSDT = pd.read_csv("BTC-USD_2022-08-01_1m.csv")
        # data_BTCUSDT = pd.DataFrame(data_BTCUSDT_yf)
        # data_BTCUSDT.to_csv("BTC-USD_2022-08-01_1m.csv")

        data_BTCUSDT_filter = data_BTCUSDT.filter(["Open"])
        # print(data_BTCUSDT_filter.shape)

        # USDT safe balance
        balance_USDT = client.get_asset_balance(asset='USDT')

        # print(balance_USDT['asset'], balance_USDT['free'], balance_USDT['locked'])

        price_balance_free_USDT = balance_USDT['free']
        price_balance_locked_USDT = balance_USDT['locked']

        # BTC safe balance
        balance_BTCUSDT = client.get_asset_balance(asset='BTC')

        # print(balance_BTCUSDT['asset'], balance_BTCUSDT['free'], balance_BTCUSDT['locked'])

        price_balance_free_BTCUSDT = balance_BTCUSDT['free']
        price_balance_locked_BTCUSDT = balance_BTCUSDT['locked']

        # ETH safe balance
        balance_ETH = client.get_asset_balance(asset='ETH')

        # print(balance_ETH['asset'], balance_ETH['free'], balance_ETH['locked'])

        price_balance_free_ETH = balance_BTCUSDT['free']
        price_balance_locked_ETH = balance_BTCUSDT['locked']

        # print(balance_btc["asset"],balance_btc["free"],balance_btc["locked"])
        balance_usd_btc_usd_free_locked_sum = float(balance_BTCUSDT["free"]) + float(balance_BTCUSDT["locked"])
        balance_usd_btc_usd_present = balance_usd_btc_usd_free_locked_sum * float(all_BTCUSDT["price"])
        # price_balance_FL_BTCUSDT = balance_BTCUSDT['free'] + balance_BTCUSDT['locked']
        # price_balance_FL_BTCUSDT = float(price_balance_free_BTCUSDT["free"]) + float(price_balance_locked_BTCUSDT["locked"])
        # balance_price_balance_FL_BTCUSDT = price_balance_FL_BTCUSDT * float(price_BTCUSDT)

        data_price_balance_locked_BTCUSDT = open("BTC-USD_2022-08-01_1m_free.csv", "a")
        data_price_balance_locked_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
        data_price_balance_locked_BTCUSDT.write(",")
        data_price_balance_locked_BTCUSDT.write(str(price_balance_free_BTCUSDT))
        data_price_balance_locked_BTCUSDT.write(",")
        data_price_balance_locked_BTCUSDT.write(str(price_balance_locked_BTCUSDT))
        data_price_balance_locked_BTCUSDT.write(",")
        data_price_balance_locked_BTCUSDT.write(str(balance_usd_btc_usd_present))
        data_price_balance_locked_BTCUSDT.write(",")
        data_price_balance_locked_BTCUSDT.write("\n")
        data_price_balance_locked_BTCUSDT.close()

        data_price_balance_locked_BTCUSDT = pd.read_csv("BTC-USD_2022-08-01_1m_free.csv")
        # data_BTCUSDT = pd.DataFrame(data_BTCUSDT_yf)
        # data_BTCUSDT.to_csv("BTC-USD_2022-08-01_1m.csv")

        data_price_balance_free_BTCUSDT_filter = data_price_balance_locked_BTCUSDT.filter(["free"])
        # print(data_price_balance_free_BTCUSDT_filter.shape)
        data_price_balance_locked_BTCUSDT_filter = data_price_balance_locked_BTCUSDT.filter(["locked"])
        # print(data_price_balance_locked_BTCUSDT_filter.shape)
        data_all_price_balance_FL_BTCUSDT_filter = data_price_balance_locked_BTCUSDT.filter(["price_usd"])
        # print(data_all_price_balance_FL_BTCUSDT_filter.shape)

        """Начало нейронной сети для BTCUSDT """

        data_BTCUSDT_filter_100 = data_BTCUSDT_filter.tail(1000)
        dataset_data_BTCUSDT_filter_100_training_data_len = math.ceil(len(data_BTCUSDT_filter_100) * .8)
        scaler_BTCUSDT_filter_100_min_max = MinMaxScaler(feature_range=(0, 1))
        scaled_data_BTCUSDT_filter_100 = scaler_BTCUSDT_filter_100_min_max.fit_transform(data_BTCUSDT_filter_100)
        # print(scaled_data_BTCUSDT_filter_100)


        # create the training dataset
        train_data_BTCUSDT = scaled_data_BTCUSDT_filter_100[0:dataset_data_BTCUSDT_filter_100_training_data_len, :]
        # split the data into x_train and y_train data sets
        x_train_BTCUSDT = []
        y_train_BTCUSDT = []
        for rar in range(60, len(train_data_BTCUSDT)):
            x_train_BTCUSDT.append(train_data_BTCUSDT[rar - 60:rar, 0])
            y_train_BTCUSDT.append(train_data_BTCUSDT[rar, 0])
            if rar <= 61:
                # print(x_train_BTCUSDT)
                # print(y_train_BTCUSDT)
                print()

        # conver the x_train and y_train to numpy arrays
        x_train_BTCUSDT, y_train_BTCUSDT = np.array(x_train_BTCUSDT), np.array(y_train_BTCUSDT)
        # reshape the data
        x_train_BTCUSDT = np.reshape(x_train_BTCUSDT, (x_train_BTCUSDT.shape[0], x_train_BTCUSDT.shape[1], 1))
        # print(x_train_BTCUSDT.shape)

        # biuld to LST model

        model_BTCUSDT = Sequential()
        model_BTCUSDT.add(LSTM(50, return_sequences=True, input_shape=(x_train_BTCUSDT.shape[1], 1)))
        model_BTCUSDT.add(LSTM(101, return_sequences=False))
        model_BTCUSDT.add(Dense(50))
        model_BTCUSDT.add(Dense(25))
        model_BTCUSDT.add(Dense(1))
        # cmopale th emodel
        model_BTCUSDT.compile(optimizer='adam', loss='mean_squared_error')
        # train_the_model
        model_BTCUSDT.summary()
        # print("Fit model on training data")

        # Evaluate the model on the test data using `evaluate`
        # print("Evaluate on test data")
        results_BTCUSDT = model_BTCUSDT.evaluate(x_train_BTCUSDT, y_train_BTCUSDT, batch_size=1)
        # print("test loss, test acc:", results_BTCUSDT)

        model_BTCUSDT = tf.keras.models.load_model(os.path.join("./dnn/", "BTC-USDT_model.h5"))
        model_BTCUSDT.fit(x_train_BTCUSDT, y_train_BTCUSDT, batch_size=1, epochs=1)

        model_BTCUSDT.save(os.path.join("./dnn/", "BTC-USDT_model.h5"))
        # reconstructed_model_BTCUSDT = tf.keras.models_BTCUSDT.load_model(os.path.join("./dnn/", "BTC-USDT_model.h5"))

        # np.testing.assert_allclose(model_BTCUSDT.predict(x_train), reconstructed_model_BTCUSDT.predict(x_train))
        # reconstructed_model_BTCUSDT.fit(x_train_BTCUSDT, y_train_BTCUSDT)

        # create the testing data set
        # create a new array containing scaled values from index 1713 to 2216
        test_data_BTCUSDT = scaled_data_BTCUSDT_filter_100[dataset_data_BTCUSDT_filter_100_training_data_len - 60:, :]
        # create the fata sets x_test and y_test
        x_test_BTCUSDT = []
        y_test_BTCUSDT = scaled_data_BTCUSDT_filter_100[dataset_data_BTCUSDT_filter_100_training_data_len:, :]
        for resr in range(60, len(test_data_BTCUSDT)):
            x_test_BTCUSDT.append(test_data_BTCUSDT[resr - 60:resr, 0])

        # conert the data to numpy array
        x_test_BTCUSDT = np.array(x_test_BTCUSDT)

        # reshape the data
        x_test_BTCUSDT = np.reshape(x_test_BTCUSDT, (x_test_BTCUSDT.shape[0], x_test_BTCUSDT.shape[1], 1))

        # get the model predicted price values
        predictions_BTCUSDT = model_BTCUSDT.predict(x_test_BTCUSDT)
        predictions_BTCUSDT = scaler_BTCUSDT_filter_100_min_max.inverse_transform(predictions_BTCUSDT)

        # get the root squared error (RMSE)
        rmse_BTCUSDT = np.sqrt(np.mean(predictions_BTCUSDT - y_test_BTCUSDT) ** 2)
        # print(rmse_BTCUSDT)

        safe_rmse_pred_price_BTCUSDT = open("rmse_BTCUSDT.csv", "a")
        safe_rmse_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
        safe_rmse_pred_price_BTCUSDT.write(",")
        safe_rmse_pred_price_BTCUSDT.write(str(rmse_BTCUSDT))
        safe_rmse_pred_price_BTCUSDT.write(",")
        safe_rmse_pred_price_BTCUSDT.write("\n")
        safe_rmse_pred_price_BTCUSDT.close()

        read_rmse_pred_price_BTCUSDT = pd.read_csv("rmse_BTCUSDT.csv")
        read_rmse_pred_price_BTCUSDT_filter = read_rmse_pred_price_BTCUSDT.filter(["rmse"])
        read_rmse_pred_price_BTCUSDT_filter_tail_10 = read_rmse_pred_price_BTCUSDT_filter.tail(1000)


        if int(rmse_BTCUSDT) >= 1000:
            from termcolor import colored

            # print(colored("Высокий уровень ошибки", "red"))

        # get the quate
        btc_quote_BTCUSDT = pd.read_csv("BTC-USD_2022-08-01_1m.csv")
        # btc_quote = pd.read_csv(str(balance_btc["asset"]) + ".csv", delimiter=",")
        # new_df = btc_quote.filter(["Well"])
        new_df_BTCUSDT = btc_quote_BTCUSDT.filter(["Open"])

        # get teh last 60 days closing price values and convert the dataframe to an array
        last_60_days_BTCUSDT = new_df_BTCUSDT[-60:].values
        # scale the data to be values beatwet 0 and 1

        last_60_days_scaled_BTCUSDT = scaler_BTCUSDT_filter_100_min_max.transform(last_60_days_BTCUSDT)

        # creAte an enemy list
        X_test_BTCUSDT = []
        # Append past 60 days
        X_test_BTCUSDT.append(last_60_days_scaled_BTCUSDT)

        # convert the x tesst dataset to numpy
        X_test_BTCUSDT = np.array(X_test_BTCUSDT)

        # Reshape the dataframe
        X_test_BTCUSDT = np.reshape(X_test_BTCUSDT, (X_test_BTCUSDT.shape[0], X_test_BTCUSDT.shape[1], 1))
        # get predict scaled

        pred_price_BTCUSDT = model_BTCUSDT.predict(X_test_BTCUSDT)
        # undo the scaling
        pred_price_BTCUSDT = scaler_BTCUSDT_filter_100_min_max.inverse_transform(pred_price_BTCUSDT)
        # print(pred_price_BTCUSDT)

        pred_price_a_BTCUSDT = pred_price_BTCUSDT[0]
        pred_price_aa_BTCUSDT = pred_price_a_BTCUSDT[0]
        preset_pred_price_BTCUSDT = str(pred_price_aa_BTCUSDT)
        # print(pred_price_BTCUSDT)
        # print(preset_pred_price_BTCUSDT)

        safe_preset_pred_price_BTCUSDT = open("preset_pred_price_BTCUSDT.csv", "a")
        safe_preset_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
        safe_preset_pred_price_BTCUSDT.write(",")
        safe_preset_pred_price_BTCUSDT.write(str(preset_pred_price_BTCUSDT))
        safe_preset_pred_price_BTCUSDT.write(",")
        safe_preset_pred_price_BTCUSDT.write("\n")
        safe_preset_pred_price_BTCUSDT.close()

        read_preset_pred_price_BTCUSDT = pd.read_csv("preset_pred_price_BTCUSDT.csv")
        read_preset_pred_price_BTCUSDT_filter = read_preset_pred_price_BTCUSDT.filter(["pred_price"])
        read_preset_pred_price_BTCUSDT_filter_tail_10 = read_preset_pred_price_BTCUSDT_filter.tail(1000)


        """Конец нейронной сети для BTCUSDT """
        # Выставление ордера

        symbol_BTCUSDT = info[11]
        if preset_pred_price_BTCUSDT <= symbol_BTCUSDT['price']:
            info = client.get_all_tickers()
            symbol_BTCUSDT = info[11]
            a = float(1)
            b = float(balance_USDT['free'])
            ab_sum = a * b
            data_coin = float(ab_sum) - min_lovume_usdt
            # print(data_coin)

            if data_coin <= 0:
                # print([data_coin, a, b])
                # print(ab_sum)
                quantity = float(min_lovume_usdt / float(symbol_BTCUSDT['price']))
                # print(quantity)
                # print("BUY USDT")
                safe_rmse_pred_price_BTCUSDT = open("order_BTCUSDT.csv", "a")
                safe_rmse_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write(str("DO NOT BUY USDT " + str(preset_pred_price_BTCUSDT)))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write("\n")
                safe_rmse_pred_price_BTCUSDT.close()

            elif data_coin >= 0:
                # print([data_coin, a, b])
                # print("\n" + "BUY USDT " + str(preset_pred_price_BTCUSDT))
                # print(a)
                quantity = float(min_lovume_usdt / float(symbol_BTCUSDT['price']))
                quantity_start = round(quantity, 5)
                # print(quantity_start)
                order = client.order_limit_buy(
                    symbol='BTCUSDT',
                    quantity=quantity_start,
                    price=round(float(preset_pred_price_BTCUSDT), 1))
                # print(order)
                safe_rmse_pred_price_BTCUSDT = open("order_BTCUSDT.csv", "a")
                safe_rmse_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write(str(order))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write("\n")
                safe_rmse_pred_price_BTCUSDT.close()




        elif preset_pred_price_BTCUSDT >= symbol_BTCUSDT['price']:
            info = client.get_all_tickers()
            symbol_BTCUSDT = info[11]
            a = float(symbol_BTCUSDT['price'])
            b = float(balance_BTCUSDT["free"])
            ab_sum = a * b
            data_coin = float(ab_sum) - min_lovume_usdt
            # print(data_coin)

            if data_coin <= 0:
                # print([data_coin, a, b])
                # print(ab_sum)
                quantity = float(min_lovume_usdt / float(symbol_BTCUSDT['price']))
                # print(quantity)
                # print("SELL BTC")
                safe_rmse_pred_price_BTCUSDT = open("order_BTCUSDT.csv", "a")
                safe_rmse_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write(str("DO NOT SELL BTC " + str(preset_pred_price_BTCUSDT)))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write("\n")
                safe_rmse_pred_price_BTCUSDT.close()

            elif data_coin >= 0:
                # print([data_coin, a, b])
                # print("\n" + "SELL BTC " + str(preset_pred_price_BTCUSDT))
                # print(a)
                quantity = float(min_lovume_usdt / float(symbol_BTCUSDT['price']))
                quantity_start = round(quantity, 5)
                # print(quantity_start)
                order = client.order_limit_sell(symbol='BTCUSDT',
                                                quantity=quantity_start,
                                                price=round(float(preset_pred_price_BTCUSDT), 1))
                # print(order)
                safe_rmse_pred_price_BTCUSDT = open("order_BTCUSDT.csv", "a")
                safe_rmse_pred_price_BTCUSDT.write(str(time.strftime("%Y-%m-%d %H:%M:%S+00:00")))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write(str(order))
                safe_rmse_pred_price_BTCUSDT.write(",")
                safe_rmse_pred_price_BTCUSDT.write("\n")
                safe_rmse_pred_price_BTCUSDT.close()


    if time_minute == "00":
        neurostart()
    elif time_minute == "15":
        neurostart()
    elif time_minute == "30":
        neurostart()
    elif time_minute == "45":
        neurostart()

    elif time_minute != ["00","15","30","45"]:
        time.sleep(1)
        print("Все нормально, ждем")
        #print(["по времени записи : ", time_minute, time.strftime("%M %S")])

