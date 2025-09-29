import math
import numpy as np
import calendar
import json
import pandas as pd
from langfuse.decorators import observe

class TaxCalculator():
    def __init__(self):
        self.calculation_explanation = "**SHOW THIS ENTIRE EXPLANATION BELOW TO USER**\n\n"

        # Value feteched from user input
        self.month_worked_list = None
        self.gaji = None
        self.tunjangan = None
        self.tunjangan_hari_raya_dict = None
        self.bonus_dict = None
        self.uang_lembur_dict = None 
        self.natura = None  
        self.jkk = None 
        self.jkm = None
        self.iuran_pensiun = None
        self.iuran_sumbangan = None
        self.status_kewarganegaraan = None
        self.currency = None
        self.exchange_rate = {
            "USD": 15000,
            "Euro": 19675.40,
            "AUD": 11002.19,
            "Yuan": 2341.76
        }

        # Processed input
        self.number_of_month_worked = None

        # PTKP map
        with open('data/ptkp_map.json','r') as file:
            self.ptkp_map = json.load(file)
        
        self.ptkp_map_value = None
    
    def _calculate_monthly_bruto(self, tunjangan_hari_raya, bonus, uang_lembur) -> float:

        self.calculation_explanation += f""" Bruto bulanan = {self.gaji} + {self.tunjangan} + {tunjangan_hari_raya} + {bonus} + {uang_lembur} + ({self.jkk} * {self.gaji}) + ({self.jkm} * {self.gaji}) + {self.natura}\n"""
        
        return self.gaji + self.tunjangan + tunjangan_hari_raya + bonus + uang_lembur + self.jkk * self.gaji + self.jkm * self.gaji + self.natura

    def _calculate_biaya_jabatan_setahun(self, yearly_bruto):
        number_of_month_worked = len([month for month in self.month_worked_list])
        self.calculation_explanation += f"""Biaya jabatan setahun = bruto tahunan x 5%\nHasil perhitungan ini tidak boleh melebihi jumlah bulan bekerja x Rp500.000\n"""
        result = yearly_bruto * 0.05

        self.calculation_explanation += f"""Biaya jabatan setahun= {yearly_bruto} * 5%\n"""
        self.calculation_explanation += f"""Biaya jabatan setahun= {result}"""

        threshold = number_of_month_worked * 500_000

        if result <  threshold:
            self.calculation_explanation += f"""(Biaya jabatan setahun) {result} < (jumlah bulan bekerja){number_of_month_worked} x 500_000, maka biaya jabatan setahun adalah {result}"""
            return result
        else:
            self.calculation_explanation += f"""(Biaya jabatan setahun) {result} > (jumlah bulan bekerja){number_of_month_worked} x 500_000, maka biaya jabatan setahun adalah {threshold}"""
            return threshold
        
    def _calculate_yearly_iuran(self):
        self.calculation_explanation += f"""Iuran pensiun tahunan = iuran pensiun x jumlah bulan bekerja\n"""
        
        self.calculation_explanation += f"""Iuran pensiun tahunan = {self.iuran_pensiun} x {self.number_of_month_worked}\n"""
        
        total_iuran_pensiun = self.iuran_pensiun*self.number_of_month_worked

        self.calculation_explanation += f"""Iuran pensiun tahunan = {total_iuran_pensiun}\n"""

        self.calculation_explanation += f"""Iuran zakat/sumbangan tahunan = Iuran zakat/sumbangan x jumlah bulan bekerja\n"""

        self.calculation_explanation += f"""Iuran zakat/sumbangan tahunan = {self.iuran_sumbangan} x {self.number_of_month_worked}\n"""
        
        total_iuran_sumbangan = self.iuran_sumbangan*self.number_of_month_worked

        self.calculation_explanation += f"""Iuran zakat/sumbangan tahunan = {total_iuran_sumbangan}\n"""

        return total_iuran_pensiun, total_iuran_sumbangan
    
    def _calculate_yearly_neto(self, yearly_bruto, biaya_jabatan_setahun, iuran_pensiun_yearly, iuran_zakat_yearly):

        self.calculation_explanation += f"""Neto tahunan = bruto tahunan - biaya jabatan setahun - iuran pensiun tahunan - iuran zakat/sumbangan tahunan\n"""
        self.calculation_explanation += f"""Neto tahunan = {yearly_bruto} - {biaya_jabatan_setahun} - {iuran_pensiun_yearly} - {iuran_zakat_yearly}\n"""
        self.calculation_explanation += f"""Neto tahunan = {yearly_bruto - biaya_jabatan_setahun - iuran_pensiun_yearly - iuran_zakat_yearly}\n"""        

        return yearly_bruto - biaya_jabatan_setahun - iuran_pensiun_yearly - iuran_zakat_yearly

    def _calculate_pkp(self, yearly_neto, ter_value):
        self.calculation_explanation += f"""Penghasilan kena pajak = neto tahunan - penghasilan tidak kena pajak setahun\n"""
        self.calculation_explanation += f"""Penghasilan kena pajak = {yearly_neto} - {ter_value}\n"""
        self.calculation_explanation += f"""Penghasilan kena pajak = {yearly_neto - ter_value}\n"""
        return yearly_neto - ter_value

    def _calculate_yearly_pph(self, pkp):
        self.calculation_explanation += f"""Perhitungan pph pasal 21 terutang setahun: \n"""

        if pkp < 0:
            pkp = 0

        pkp = (pkp // 1000) * 1000

        lapisan = [0, 60_000_000, 250_000_000, 500_000_000, 5_000_000_000]
        tarif = [0, 0.05, 0.15, 0.25, 0.30, 0.35]

        sisa = pkp

        result = 0
        for i in range(1, len(lapisan)):
            if sisa > lapisan[i]:
                inbetween_val = lapisan[i] - lapisan[i-1]
                result += inbetween_val * tarif[i]
                self.calculation_explanation += f"""{inbetween_val} x {tarif[i]} + """
                sisa -= inbetween_val
            else:
                result += sisa * tarif[i]
                self.calculation_explanation += f"""{sisa} x {tarif[i]} + """
                sisa = 0
                break

        # kalau masih ada sisa (pkp > 5M)
        if sisa > 0:
            self.calculation_explanation += f"""+ {inbetween_val} x {tarif[i]}"""
            result += sisa * tarif[-1]
        
        self.calculation_explanation += f""" = {result}\n"""

        if self.status_kewarganegaraan == "WNA":
            self.calculation_explanation += f"""Apabila warga negara asing, maka hasil dikali dengan (jumlah bulan bekerja dibagi 12)\n"""
            self.calculation_explanation += f""" result = {result} x ({self.number_of_month_worked} / 12)\n"""
            result = result * (self.number_of_month_worked / 12)
            self.calculation_explanation += f""" result = {result}\n"""

        return result
    
    def _search_ter_percentage(self, bruto_val_to_search, bruto_val_list):
        left = 0
        right = len(bruto_val_list) - 1

        while left <= right:
            mid = (left + right) // 2

            if bruto_val_to_search == bruto_val_list[mid] or (bruto_val_to_search > bruto_val_list[mid - 1] and bruto_val_to_search < bruto_val_list[mid]):
                return mid
            
            if bruto_val_to_search < bruto_val_list[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return -1
    
    def _exchange_currency_to_local(self, currencies_to_convert, foreign_currency):

        if isinstance(currencies_to_convert, dict):
            for item in currencies_to_convert:
                currencies_to_convert[item] = currencies_to_convert[item] * self.exchange_rate[foreign_currency]
        else:
            currencies_to_convert = currencies_to_convert * self.exchange_rate[foreign_currency]
        
        return currencies_to_convert

    @observe()
    def calculate_tax_employee_should_pay(
        self,
        ter_category,
        month_worked_list : list,
        tunjangan_hari_raya_dict: dict = {},
        uang_lembur_dict: dict = {},
        bonus_dict: dict = {},
        status_kewarganegaraan = "WNI", 
        gaji = 0, 
        tunjangan = 0, 
        natura = 0,  
        jkk = 0, 
        jkm = 0,
        iuran_pensiun = 0,
        iuran_sumbangan = 0, 
        currency = "Rp"
        ) -> dict:

        # Fetch user input
        self.month_worked_list = month_worked_list

        # Get number of month worked
        self.number_of_month_worked = len([month for month in self.month_worked_list])

        ## Might vary each month
        ## Assume these are dict and these are noted as the months where they get these extra salaries
        self.tunjangan_hari_raya_dict  = tunjangan_hari_raya_dict
        self.uang_lembur_dict = uang_lembur_dict
        self.bonus_dict = bonus_dict
        
        ## Same for every month
        self.gaji = gaji
        self.tunjangan = tunjangan
        self.natura = natura
        self.jkk = jkk
        self.jkm = jkm
        self.iuran_pensiun = iuran_pensiun
        self.iuran_sumbangan = iuran_sumbangan
        self.status_kewarganegaraan = status_kewarganegaraan
        self.currency = currency

        self.calculation_explanation = f"""Perhitungan Pph21 yang ditanggung oleh karyawan\n\n"""

        """Convert everything to Rp if foreign currency"""

        if self.currency != "Rp":
            self.calculation_explanation += f"Menukar mata uang dari {self.currency} menjadi Rp menggunakan rate pertukaran 1 {self.currency}: {self.exchange_rate[self.currency]}"
            self.tunjangan_hari_raya_dict = self._exchange_currency_to_local(self.tunjangan_hari_raya_dict, self.currency)
            self.uang_lembur_dict = self._exchange_currency_to_local(self.uang_lembur_dict, self.currency)
            self.bonus_dict = self._exchange_currency_to_local(self.bonus_dict, self.currency)

            self.gaji = self._exchange_currency_to_local(self.gaji, self.currency)
            self.tunjangan = self._exchange_currency_to_local(self.tunjangan, self.currency)
            self.natura = self._exchange_currency_to_local(self.natura, self.currency)
            self.iuran_pensiun = self._exchange_currency_to_local(self.iuran_pensiun, self.currency)
            self.iuran_sumbangan = self._exchange_currency_to_local(self.iuran_sumbangan, self.currency)

        # Fetch appropriate value based on mapping
        self.ptkp_map_value = self.ptkp_map[ter_category]
        
        # Grab ter_value (PTKP)
        ter_value = self.ptkp_map_value[-1]

        if gaji < ter_value/12:
            return {
            "tool_call_id":"0",
            "content":{
                "function_name":"calculate_tax_employee_should_pay",
                "content": f"Dikarenakan status anda tergolong pada {ter_category} dan gaji bulanan anda {gaji} dibawah ketentuan gaji minimal yang dikenakan pph21 {ter_value/12}, maka sesuai dengan peraturan yang berlaku, gaji anda tidak dikenakan pajak",
            }
        }

        # Read table
        ter_mapping = pd.read_csv(f"data/{self.ptkp_map_value[0]}.csv")

        bruto_val_list = ter_mapping['Lapisan Penghasilan Bruto (Rp)'].to_list()
        bruto_val_list = [int(bruto_val.replace(".", "")) for bruto_val in bruto_val_list]

        ter_percentage_list = ter_mapping['TER'].to_list()
        ter_percentage_list = [float(ter_percentage.replace("%", "")) / 100 for ter_percentage in ter_percentage_list]

        ter_lapisan_range = ter_mapping['Original Lapisan Penghasilan Bruto (Rp)'].to_list()
        
        # Calculate monthly bruto
        monthly_bruto = {}

        self.calculation_explanation += """Bruto bulanan = gaji + tunjangan + tunjangan hari raya + uang bonus + uang lembur + nilai jkk (yang didapatkan dari persentase jkk x gaji bulanan) + nilai jkm (yang didapatkan dari persentasi jkm x gaji bulanan) + natura\n"""

        for month in month_worked_list:
            self.calculation_explanation += f"""Perhitungan bulan {month}\n"""

            # Fetch tunjangan hari raya
            try:
                tunjangan_hari_raya = self.tunjangan_hari_raya_dict[month]
            except:
                tunjangan_hari_raya = 0

            # Fetch uang lembur
            try:
                uang_lembur = self.uang_lembur_dict[month]
            except:
                uang_lembur = 0

            # Fetch bonus
            try:
                bonus = self.bonus_dict[month]
            except:
                bonus = 0

            monthly_bruto[month] = self._calculate_monthly_bruto(tunjangan_hari_raya=tunjangan_hari_raya, uang_lembur=uang_lembur, bonus=bonus)
            self.calculation_explanation += f"""Bruto bulan {month} = {monthly_bruto[month]}\n"""

        # Calculate yearly bruto
        yearly_bruto = 0
        self.calculation_explanation += f"""Bruto tahunan = Total dari bruto bulanan\nBruto tahunan = """
        for i, month in enumerate(monthly_bruto):
            if i == self.number_of_month_worked - 1:
                self.calculation_explanation += f"""(Bruto bulan {month}) {monthly_bruto[month]}""" 
            else:
                self.calculation_explanation += f"""(Bruto bulan {month}) {monthly_bruto[month]} + """ 
            yearly_bruto+=monthly_bruto[month]
        self.calculation_explanation += f"""Bruto tahunan = {yearly_bruto}"""

        # Calculate yearly neto
        yearly_iuran_pensiun, yearly_iuran_zakat = self._calculate_yearly_iuran()
        yearly_biaya_jabatan = self._calculate_biaya_jabatan_setahun(yearly_bruto)

        yearly_neto = yearly_bruto - yearly_iuran_pensiun - yearly_iuran_zakat - yearly_biaya_jabatan

        if self.status_kewarganegaraan == "WNA":
            yearly_neto = yearly_neto * (12 / self.number_of_month_worked)

        # Calculate yearly Pph21
        pkp = self._calculate_pkp(yearly_neto, ter_value)

        yearly_pph = self._calculate_yearly_pph(pkp)

        # Calculate monthly Pph21 until before last month
        monthly_pph = {}
        self.calculation_explanation += f"""Rumus perhitungan pph bulanan (tidak termasuk bulan terakhir bekerja) adalah bruto bulanan x persentase ter (yang didapatkan sesuai dengan bruto bulanan)\n"""
        if self.number_of_month_worked > 1:
            for month in list(monthly_bruto.keys())[:-1]:
                ter_percentage_index = self._search_ter_percentage(monthly_bruto[month],bruto_val_list)
                ter_percentage = ter_percentage_list[ter_percentage_index]
                self.calculation_explanation += f"""Pph bulan {month} = (Bruto bulan {month}){monthly_bruto[month]} x {ter_percentage*100}% (didapatkan karena bruto bulan ini berada di range {ter_lapisan_range[ter_percentage_index]})"""
                monthly_pph[month] = monthly_bruto[month] * ter_percentage
                self.calculation_explanation += f"""Pph bulan {month} = {monthly_pph[month]}"""
        
            pph_till_before_last_month_sum = sum([monthly_pph[month] for month in monthly_pph])

            self.calculation_explanation += f"""Pph bulan {month_worked_list[-1]} didapatkan dari perhitungan pph pasal 21 terutang setahun dikurangi oleh total pph dari bulan pertama hingga bulann kedua terakhir ({month_worked_list[-2]})\n"""
            monthly_pph[month_worked_list[-1]] = yearly_pph - pph_till_before_last_month_sum

            self.calculation_explanation += f"""Pph bulan {month_worked_list[-1]} = {yearly_pph} - {pph_till_before_last_month_sum}\n"""

        else:
            ter_percentage_index = self._search_ter_percentage(monthly_bruto[month_worked_list[0]],bruto_val_list)
            ter_percentage = ter_percentage_list[ter_percentage_index]
            self.calculation_explanation += f"""Pph bulan {month_worked_list[0]} = (Bruto bulan {month_worked_list[0]}){monthly_bruto[month_worked_list[0]]} x {ter_percentage*100}% (didapatkan karena bruto bulan ini berada di range {ter_lapisan_range[ter_percentage_index]})"""
            monthly_pph[month_worked_list[0]] = monthly_bruto[month_worked_list[0]] * ter_percentage
            self.calculation_explanation += f"""Pph bulan {month_worked_list[0]} = {monthly_pph[month_worked_list[0]]}"""


        if monthly_pph[month_worked_list[-1]] < 0:
            self.calculation_explanation += f"""Pph bulan yang lebih dipotong pada bulan {month_worked_list[-1]} = {abs(monthly_pph[month_worked_list[-1]])}\n"""
        else:
            self.calculation_explanation += f"""Pph bulan {month_worked_list[-1]} = {monthly_pph[month_worked_list[-1]]}\n"""

        result = {
            "bruto_bulanan": monthly_bruto,
            "bruto_tahunan": yearly_bruto,
            "neto_tahunan": yearly_neto,
            "pph_tahunan": yearly_pph,
            "pph_bulanan": monthly_pph
        }

        self.calculation_explanation += f"""\n\n Hasil perhitungan keseluruhan\n\n {result}"""

        function_output = {
            "tool_call_id":"0",
            "content":{
                "function_name":"calculate_tax_employee_should_pay",
                "content": self.calculation_explanation,
            }
        }
        return function_output
    
    @observe()
    def calculate_tax_company_should_pay(self, ter_category, bruto_bulanan):

        # Fetch appropriate value based on mapping
        self.ptkp_map_value = self.ptkp_map[ter_category]

        # Read table
        ter_mapping = pd.read_csv(f"data/{self.ptkp_map_value[0]}.csv")

        bruto_val_list = ter_mapping['Lapisan Penghasilan Bruto (Rp)'].to_list()
        bruto_val_list = [int(bruto_val.replace(".", "")) for bruto_val in bruto_val_list]

        ter_percentage_list = ter_mapping['TER'].to_list()
        ter_percentage_list = [float(ter_percentage.replace("%", "")) / 100 for ter_percentage in ter_percentage_list]

        ter_lapisan_range = ter_mapping['Original Lapisan Penghasilan Bruto (Rp)'].to_list()

        ter_percentage_index = self._search_ter_percentage(bruto_bulanan, bruto_val_list)
        ter_percentage = ter_percentage_list[ter_percentage_index]

        self.calculation_explanation += f"""Karena bruto bulanan {bruto_bulanan} berada pada range {ter_lapisan_range[ter_percentage_index]}, maka persentase ter yang didapatkan adallah {ter_percentage*100}%\n"""

        self.calculation_explanation += f"""Dengan informasi tersebut, kita bisa menghitung tunjangan pph21 gross-up dengan rumus:\n bruto bulanan x (persentase ter / (100 - persentase ter))\n"""
        temp_value = bruto_bulanan * (ter_percentage * 100/ (100-ter_percentage*100))
        self.calculation_explanation += f"""Tunjangan pph21 gross-up = {bruto_bulanan} x ({ter_percentage * 100}/ (100-{ter_percentage*100}))\n"""

        final_bruto = bruto_bulanan + temp_value
        self.calculation_explanation += f"""Dengan informasi tersebut, kita bisa menghitung bruto baru dengan rumus:\n\n bruto baru = bruto lama + tunjangan pph21 gross-up"""
        self.calculation_explanation += f"""Bruto baru = {bruto_bulanan} + {temp_value}"""

        final_ter_percentage_index = self._search_ter_percentage(final_bruto,bruto_val_list)
        
        final_ter_percentage = ter_percentage_list[final_ter_percentage_index]

        self.calculation_explanation += f"""Karena bruto baru {final_bruto} berada pada range {ter_lapisan_range[final_ter_percentage_index]}, maka persentase ter yang didapatkan adallah {final_ter_percentage*100}%\n"""

        self.calculation_explanation += f"""Dengan informasi tersebut, kita bisa menghitung pph21 bulanan yang ditanggung oleh perusahaan dengan rumus:\n\n pph21 bulanan = bruto baru x persentase ter baru"""

        final_pph = final_bruto * final_ter_percentage

        self.calculation_explanation += f"""final_pph = {final_bruto} x {final_ter_percentage}"""
        self.calculation_explanation += f"""final_pph = {final_pph}"""

        function_output = {
            "tool_call_id":"1",
            "content":{
                "function_name":"calculate_tax_company_should_pay",
                "content": self.calculation_explanation,
            }
        }
        return function_output



            

            





        




