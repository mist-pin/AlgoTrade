import MetaTrader5 as mt5
from datetime import datetime, timedelta
import asyncio
import aiofiles

import pandas
import pandas as pd
import os
from aiofiles import os as aios
from pandas import DataFrame

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def setup()->bool:
    """
    Provides functionalities for initializing MetaTrader 5 (MT5) terminal, checking the connection,
    and confirming the account information for financial trading operations.

    Functions
    ---------

    function:: initializer()

       Initializes the MetaTrader 5 platform. Ensures the terminal is properly configured before
       performing any trading or data-fetching tasks.

    :return: True if MT5 initialization is successful, otherwise False.

    function:: account_info()

       Fetches and displays account information to confirm successful connection to the MetaTrader 5
       account. Also verifies if the account is accessible or alerts in case of connection failure.

    :return: True if connected successfully, otherwise False.
    """
    def initializer():
        if not mt5.initialize():
            print("Failed to initialize MetaTrader 5")
            return False
        else:
            print('Initialized successfully')
            return True

    def account_info():
        # Display account information to confirm connection
        account_data = mt5.account_info()
        if account_data is None:
            print("Failed to connect to the account. Check login details.")
            mt5.shutdown()
            return False
        print(f"Connected to account {account_data.login} on server {account_data.server}")
        return True

    # Initialize MT5 and server connection
    if all([initializer(), account_info()]):
        return True
    else:return False


async def get_recent_data(symbol:str, timeframe:int=1)-> tuple[DataFrame, DataFrame]:
    """
    Fetches recent market data for a specified symbol and timeframe, processes the data, and returns it
    as a tuple of DataFrame objects or a single DataFrame object. The function retrieves tick data using
    MetaTrader 5, processes the retrieved data into a DataFrame, and optionally formats them into a
    second DataFrame summarizing information such as high, low, open, close, and volume within the given
    timeframe. The processed data is written to a CSV file asynchronously.

    :param symbol: The financial instrument symbol for which to fetch recent market data.
    :type symbol: str
    :param timeframe: The timeframe in seconds to fetch recent market data. Defaults to 1 second.
    :type timeframe: int
    :return: A tuple containing the complete tick DataFrame and summarized DataFrame if data exists else None.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    now = datetime.now()
    previous_sec = now - timedelta(seconds=timeframe)
    data = mt5.copy_ticks_range(symbol, previous_sec, now, mt5.COPY_TICKS_ALL)
    data = pd.DataFrame(data)
    await to_csv_file(data)
    data['time'] = pd.to_datetime(data['time_msc'], unit='ms')
    data = data.drop(['time_msc','last', 'volume', 'flags', 'volume_real'], axis=1)

    sec_data  = None
    if len(data) > 0:
        sec_data = pd.DataFrame(
            [[
                data['time'].iloc[0].replace(microsecond=0),
                data['ask'].max(),
                data['bid'].min(),
                data['bid'].iloc[0],
                data['ask'].iloc[-1],
                data['bid'].count()
            ]],
            columns=['time', 'high', 'low', 'open', 'close', 'volume']
        )
        await to_csv_file(sec_data, filename='sec_data.csv')
    return data, sec_data


async def to_csv_file(dataframe:pd.DataFrame, filename:str ='tick_data.csv')->None:
    """
    Write the content of a pandas DataFrame to a CSV file asynchronously. This function ensures
    that a designated directory exists for storing the file, appends to the file if it
    already exists depending on the 'new' parameter, or overwrites/recreates the file if
    specified. The function is designed for efficient non-blocking I/O operations.

    :param dataframe: The pandas DataFrame to be written into a CSV file.
    :type dataframe: pd.DataFrame
    :param filename: The name of the file to write to. Defaults to 'tick_data.csv'.
    :type filename: str, optional
    :return: This function does not return a value; its sole purpose is to write
        data to the designated CSV file.
    :rtype: None
    """
    path = os.path.join(os.getcwd(), 'data_set') + os.sep
    file = os.path.join(path, filename)
    if await aios.path.exists(file):
        f = await aiofiles.open(file, 'a')
        await f.write(dataframe.to_csv(header=False, index=False, lineterminator='\n'))
        await f.close()
    else:
        f = await aiofiles.open(file, 'w')
        await f.write(dataframe.to_csv(index=False, lineterminator='\n'))
        await f.close()


class TechnicalAnalysis:
    def __init__(self):
        """Initialize the TechnicalAnalysis class with previous values for real-time calculations."""
        self.prev_close = None
        self.prev_ema = {}
        self.prev_sma_values = {}
        self.prev_rsi_gain = 0
        self.prev_rsi_loss = 0
        self.prev_macd_ema_12 = None
        self.prev_macd_ema_26 = None
        self.prev_macd_signal = None
        self.prev_atr = None

    def __sma(self, close, period):
        """Calculate SMA with shared previous values across different periods."""
        if period not in self.prev_sma_values:
            self.prev_sma_values[period] = []

        self.prev_sma_values[period].append(close)

        if len(self.prev_sma_values[period]) > period:
            self.prev_sma_values[period].pop(0)

        return sum(self.prev_sma_values[period]) / len(self.prev_sma_values[period])

    def __ema(self, close, period):
        """Calculate EMA with shared previous values across different periods."""
        alpha = 2 / (period + 1)
        if period not in self.prev_ema:
            self.prev_ema[period] = close  # Initialize with first close

        self.prev_ema[period] = (alpha * close) + ((1 - alpha) * self.prev_ema[period])
        return self.prev_ema[period]

    def __rsi(self, close, period=14):
        """Calculate RSI with previous gain/loss values."""
        if self.prev_close is None:
            self.prev_close = close
            return 50  # Neutral RSI for first data point

        delta = close - self.prev_close
        gain = max(delta, 0)
        loss = abs(min(delta, 0))

        self.prev_rsi_gain = ((self.prev_rsi_gain * (period - 1)) + gain) / period
        self.prev_rsi_loss = ((self.prev_rsi_loss * (period - 1)) + loss) / period

        rs = self.prev_rsi_gain / self.prev_rsi_loss if self.prev_rsi_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        self.prev_close = close  # Update previous close
        return rsi

    def __macd(self, close):
        """Calculate MACD and Signal Line using previous EMA values."""
        self.prev_macd_ema_12 = self.__ema(close, 12) if self.prev_macd_ema_12 is None else self.__ema(close, 12)
        self.prev_macd_ema_26 = self.__ema(close, 26) if self.prev_macd_ema_26 is None else self.__ema(close, 26)
        macd = self.prev_macd_ema_12 - self.prev_macd_ema_26

        self.prev_macd_signal = self.__ema(macd, 9) if self.prev_macd_signal is None else self.__ema(macd, 9)
        return macd, self.prev_macd_signal

    def __atr(self, high, low, close, period=14):
        """Calculate ATR using previous values."""
        if self.prev_close is None:
            self.prev_close = close
            return 0  # First value is 0

        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))

        if self.prev_atr is None:
            self.prev_atr = tr
        else:
            self.prev_atr = ((self.prev_atr * (period - 1)) + tr) / period
        return self.prev_atr

    def calculate_all(self, data: dict) -> pd.DataFrame:
        """
        Compute all indicators using a single OHLCV data point.
        :param data: Dictionary with 'open', 'high', 'low', 'close', 'volume', 'time'
        :return: DataFrame with calculated indicator values
        """
        close, high, low = data['close'], data['high'], data['low']

        indicators = {
            'sma_5': self.__sma(close, 5),
            'ema_5': self.__ema(close, 5),
            'sma_10': self.__sma(close, 10),
            'ema_10': self.__ema(close, 10),
            'rsi': self.__rsi(close, 14),
            'macd': self.__macd(close)[0],
            'macd_signal': self.__macd(close)[1],
            'atr': self.__atr(high, low, close, 14)
        }

        return pd.DataFrame([indicators])


async def main(symbol = 'BTCUSD'):
    assert setup(), "Failed to setup MT5"

    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol: {symbol}")
        mt5.shutdown()
        return

    time_delay = 1
    seconds_counter = 0
    sec_data_set = pandas.DataFrame()
    analytics = TechnicalAnalysis()
    try:
        while True:
            seconds_counter += 1
            await asyncio.sleep(time_delay)
            tick = await get_recent_data(symbol)
            if tick[1] is not None and not tick[1].empty:
                sec_data = tick[1]
                analysed_data = analytics.calculate_all(
                    {'low': sec_data['low'].values[0], 'high': sec_data['high'].values[0],
                     'close': sec_data['close'].values[0]}
                )
                sec_data = pandas.concat([sec_data, analysed_data], axis=1)
                sec_data_set = pandas.concat([sec_data_set, sec_data])
                print(sec_data_set)



    except KeyboardInterrupt and asyncio.exceptions.CancelledError:
        print("Data fetching stopped by user")
        mt5.shutdown()


asyncio.run(main())
