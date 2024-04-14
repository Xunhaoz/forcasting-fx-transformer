from enum import Enum
import warnings

import pandas as pd

warnings.filterwarnings('ignore')


class Nation(Enum):
    AUSTRALIA = {
        'FX': 'AUSTRALIAN $ TO US $ (RFV) - EXCHANGE RATE',
        'FFX': 'US $ TO AUSTRALIAN $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'AUSTRALIAN DOLLAR 3M DEPOSIT (RFV) - MIDDLE RATE',
    }

    AUSTRIA = {
        'FX': 'AUSTRIAN SCHIL.TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'AUSTRIAN TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'OE 3M VIBOR DELAYED SEE EIBOR3M - OFFERED RATE',
    }

    BELGIUM = {
        'FX': 'BELGIAN FRANC TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'BELGIUM TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'BG 3M INTBK DELAYED SEE EIBOR3M - OFFERED RATE',
    }

    CANADA = {
        'FX': 'CANADIAN $ TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'CANADIAN $ TO US $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'CAN 3M T-BILL (USE CNTBB3M) - MIDDLE RATE',
    }

    DENMARK = {
        'FX': 'DANISH KRONE TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'DANISH KRONE TO US $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'DENMARK INTERBANK 3M DELAYED - OFFERED RATE',
    }

    # EUROPE = {
    #     'FX': 'EURO TO US $ (RFV) - EXCHANGE RATE',
    #     'FFX': 'US $ TO EURO 1M FWD (RFV) - EXCHANGE RATE',
    #     'INTEREST': '',
    # }

    FINLAND = {
        'FX': 'FINNISH MARKKA TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'FINLAND TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'FN 3M INTBK DELAYED SEE EIBOR3M - OFFERED RATE',
    }

    FRANCE = {
        'FX': 'FRENCH FRANC TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'FRANCE TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'FRANCE TREASURY BILL 3 MONTHS - BID RATE',
    }

    GERMANY = {
        'FX': 'GERMAN MARK TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'GERMANY TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'BD 3M INTBK DELAYED SEE EIBOR3M - OFFERED RATE',
    }

    ITALY = {
        'FX': 'ITALIAN LIRA TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'ITALY TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'ITALY T-BILL AUCT. GROSS 3 MONTH - MIDDLE RATE',
    }

    JAPAN = {
        'FX': 'JAPANESE YEN TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'JAPANESE YEN TO US $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'JAPAN INTERBANK JPY 3M - OFFERED RATE',
    }

    NORWAY = {
        'FX': 'NORWEGIAN KRONE TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'NORWAY KRONE TO US $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'NORWAY INTERBANK 3 MONTH DELAYED - OFFERED RATE',
    }

    SOUTH_KOREA = {
        'FX': 'SOUTH KOREAN WON TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'KOREA TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'SOUTH KOREA IBK. 3M SEOUL - OFFERED RATE',
    }

    SWEDEN = {
        'FX': 'SWEDISH KRONA TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'SWEDEN TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'SWEDEN IBK. STIBOR 3M DELAYED - MIDDLE RATE',
    }

    SWITZERLAND = {
        'FX': 'SWISS FRANC TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'SWISS FRANC TO US $ 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'SWISS FRANC 3M DEPOSIT (FT/RFV) - MIDDLE RATE',
    }

    TAIWAN = {
        'FX': 'TAIWAN NEW $ TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'TWD TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'TAIWAN MONEY MARKET 90 DAYS - MIDDLE RATE',
    }

    SPAIN = {
        'FX': 'SPANISH PESETA TO US $ (WMR) - EXCHANGE RATE',
        'FFX': 'SPAIN TO USD 1M FWD OR (WMR) - EXCHANGE RATE',
        'INTEREST': 'SPAIN INTERBANK 3M - MIDDLE RATE',
    }

    UNITED_KINGDOM = {
        'FX': 'UK  TO US',
        'FFX': 'US $ TO US GBP 1M FWD (RFV) - EXCHANGE RATE',
        'INTEREST': 'UK TREASURY BILL TENDER 3M - MIDDLE RATE',
    }

    US = {
        'FX': '',
        'FFX': '',
        'INTEREST': 'IBA USD IBK. LIBOR 3M DELAYED - OFFERED RATE',
    }


class DataLoader:
    def __init__(self, nation: Nation):
        self.nation = nation
        self.fx, self.ffx, self.interest = self.load_data()

    def load_data(self):
        fx = pd.read_csv('exchange_rate.csv')
        ffx = pd.read_csv('1M_FORWARD_RATE.csv')
        nation_interest = pd.read_csv('interest_rate.csv')
        us_interest = pd.read_csv('interest_rate(US).csv')

        fx = fx.set_index('Date')
        fx = fx.loc[fx.index.notna()]

        ffx = ffx.set_index('Date')
        ffx = ffx.loc[ffx.index.notna()]

        nation_interest = nation_interest.set_index('Date')
        nation_interest = nation_interest.loc[nation_interest.index.notna()]

        us_interest = us_interest.set_index('Date')
        us_interest = us_interest.loc[us_interest.index.notna()]

        interest = pd.concat([nation_interest, us_interest], axis=1)

        return fx, ffx, interest

    def get_data(self):
        nation_fx = self.fx[self.nation.value['FX']]

        nation_ffx = self.ffx[self.nation.value['FFX']]
        if self.nation in [Nation.AUSTRALIA, Nation.UNITED_KINGDOM]:
            nation_ffx = 1 / nation_ffx

        nation_interest = self.interest[self.nation.value['INTEREST']]
        us_interest = self.interest['IBA USD IBK. LIBOR 3M DELAYED - OFFERED RATE']
        df = pd.concat([nation_fx, nation_ffx, nation_interest, us_interest], axis=1)
        df.columns = ['FX', 'FFX', 'INTEREST', 'US_INTEREST']

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df


australia = DataLoader(Nation.AUSTRALIA).get_data()
austria = DataLoader(Nation.AUSTRIA).get_data()
belgium = DataLoader(Nation.BELGIUM).get_data()
canada = DataLoader(Nation.CANADA).get_data()
denmark = DataLoader(Nation.DENMARK).get_data()
finland = DataLoader(Nation.FINLAND).get_data()
france = DataLoader(Nation.FRANCE).get_data()
germany = DataLoader(Nation.GERMANY).get_data()
italy = DataLoader(Nation.ITALY).get_data()
japan = DataLoader(Nation.JAPAN).get_data()
norway = DataLoader(Nation.NORWAY).get_data()
south_korea = DataLoader(Nation.SOUTH_KOREA).get_data()
sweden = DataLoader(Nation.SWEDEN).get_data()
switzerland = DataLoader(Nation.SWITZERLAND).get_data()
taiwan = DataLoader(Nation.TAIWAN).get_data()
spain = DataLoader(Nation.SPAIN).get_data()
uk = DataLoader(Nation.UNITED_KINGDOM).get_data()

australia['2004-07-26': '2024-04-05'].to_csv('../nations/australia.csv')
austria['2004-07-26': '2024-04-05'].to_csv('../nations/austria.csv')
belgium['2004-07-26': '2024-04-05'].to_csv('../nations/belgium.csv')
canada['2004-07-26': '2024-04-05'].to_csv('../nations/canada.csv')
denmark['2004-07-26': '2024-04-05'].to_csv('../nations/denmark.csv')
finland['2004-07-26': '2024-04-05'].to_csv('../nations/finland.csv')
france['2004-07-26': '2024-04-05'].to_csv('../nations/france.csv')
germany['2004-07-26': '2024-04-05'].to_csv('../nations/germany.csv')
italy['2004-07-26': '2024-04-05'].to_csv('../nations/italy.csv')
japan['2004-07-26': '2024-04-05'].to_csv('../nations/japan.csv')
norway['2004-07-26': '2024-04-05'].to_csv('../nations/norway.csv')
south_korea['2004-07-26': '2024-04-05'].to_csv('../nations/south_korea.csv')
sweden['2004-07-26': '2024-04-05'].to_csv('../nations/sweden.csv')
switzerland['2004-07-26': '2024-04-05'].to_csv('../nations/switzerland.csv')
taiwan['2004-07-26': '2024-04-05'].to_csv('../nations/taiwan.csv')
spain['2004-07-26': '2024-04-05'].to_csv('../nations/spain.csv')
uk['2004-07-26': '2024-04-05'].to_csv('../nations/uk.csv')
