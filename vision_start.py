import time
import tkinter as tk
from tkinter import *
from tkinter import LabelFrame
from tkinter import Button
from tkinter import Entry
weight, height = 800,600
ws = tk.Tk()
ws.title('Python_dikry_binance')
ws.geometry(str(weight) + 'x' + str(height))


label = tk.Label(
    text="Программа создания отчетов BINANCE от Дмитрия Крылосова" + "\nНажмите чтобы сделать предсказание ",
    foreground="yellow",  # Устанавливает белый текст
    background="black",  # Устанавливает черный фон
    width=50,
    height=5,
)
label.grid(row=0, column=0)

time_minute = time.strftime("%M")
while True:

    ws.update()