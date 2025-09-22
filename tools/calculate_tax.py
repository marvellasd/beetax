import math
import numpy as np
import calendar
import json
import pandas as pd

class TaxCalculator():
    def __init__(self):
        # Value feteched from user input

        # ideally is a list not an integer but for now use integer
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
        self.iuran_zakat = None

        # PTKP map
        with open('data/ptkp_map.json','r') as file:
            self.ptkp_map = json.load(file)
        
        self.ptkp_map_value = None
    
    def _calculate_monthly_bruto(self, tunjangan_hari_raya, bonus, uang_lembur) -> float:

        return self.gaji + self.tunjangan + tunjangan_hari_raya + bonus + uang_lembur + self.jkk * self.gaji + self.jkm * self.gaji + self.natura

    def _calculate_biaya_jabatan_setahun(self, yearly_bruto):
        number_of_month_worked = len([month for month in self.month_worked_list])
        result = yearly_bruto * 0.05

        threshold = number_of_month_worked * 500_000

        if result <  threshold:
            return result
        else:
            return threshold
        
    def _calculate_yearly_iuran(self):
        number_of_month_worked = len([month for month in self.month_worked_list])
        return self.iuran_pensiun*number_of_month_worked, self.iuran_zakat*number_of_month_worked
    
    def _calculate_yearly_neto(self, yearly_bruto, biaya_jabatan_setahun, iuran_pensiun_yearly, iuran_zakat_yearly):
        return yearly_bruto - biaya_jabatan_setahun - iuran_pensiun_yearly - iuran_zakat_yearly

    def _calculate_pkp(self, yearly_neto, ter_value):
        return yearly_neto - ter_value

    def _calculate_yearly_pph(self, pkp):
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
                sisa -= inbetween_val
            else:
                result += sisa * tarif[i]
                sisa = 0
                break

        # kalau masih ada sisa (pkp > 5M)
        if sisa > 0:
            result += sisa * tarif[-1]

        return result

    def _calculate_dec_pph(self, yearly_pph, monthly_pph_till_nov):
        return yearly_pph - monthly_pph_till_nov
    
    def _get_month_number(self, month_input):
        return list(calendar.month_name).index(month_input.title())
    
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

    
    def calculate_tax(
        self,
        ter_category,
        month_worked_list : list,
        tunjangan_hari_raya_dict: dict = {},
        uang_lembur_dict: dict = {},
        bonus_dict: dict = {}, 
        gaji = 0, 
        tunjangan = 0, 
        natura = 0,  
        jkk = 0, 
        jkm = 0,
        iuran_pensiun = 0,
        iuran_zakat = 0 
        ) -> dict:

        # Fetch user input
        ## Assume its jan to dec for now
        self.month_worked_list = month_worked_list

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
        self.iuran_zakat = iuran_zakat

        # Fetch appropriate value based on mapping
        self.ptkp_map_value = self.ptkp_map[ter_category]
        
        # Grab ter_value (PTKP)
        ter_value = self.ptkp_map_value[-1]

        # Read table
        ter_mapping = pd.read_csv(f"data/{self.ptkp_map_value[0]}.csv")

        bruto_val_list = ter_mapping['Lapisan Penghasilan Bruto (Rp)'].to_list()
        bruto_val_list = [int(bruto_val.replace(".", "")) for bruto_val in bruto_val_list]

        ter_percentage_list = ter_mapping['TER'].to_list()
        ter_percentage_list = [float(ter_percentage.replace("%", "")) / 100 for ter_percentage in ter_percentage_list]
        
        # Calculate monthly bruto
        monthly_bruto = {}

        for month in month_worked_list:
            
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

        # Calculate yearly bruto
        yearly_bruto = 0
        for month in monthly_bruto:
            yearly_bruto+=monthly_bruto[month]

        # Calculate yearly neto
        yearly_iuran_pensiun, yearly_iuran_zakat = self._calculate_yearly_iuran()
        yearly_biaya_jabatan = self._calculate_biaya_jabatan_setahun(yearly_bruto)

        yearly_neto = yearly_bruto - yearly_iuran_pensiun - yearly_iuran_zakat - yearly_biaya_jabatan

        # Calculate yearly Pph21
        pkp = self._calculate_pkp(yearly_neto, ter_value)

        yearly_pph = self._calculate_yearly_pph(pkp)

        # Calculate monthly Pph21 until November
        monthly_pph = {}

        for i, month in enumerate(monthly_bruto):

            if i == 11:
                break
                
            ter_percentage_index = self._search_ter_percentage(monthly_bruto[month],bruto_val_list)
            ter_percentage = ter_percentage_list[ter_percentage_index]
            monthly_pph[month] = monthly_bruto[month] * ter_percentage

        pph_till_nov_sum = sum([monthly_pph[month] for month in monthly_pph])

        monthly_pph['December'] = yearly_pph - pph_till_nov_sum

        result = {
            "bruto_bulanan": monthly_bruto,
            "bruto_tahunan": yearly_bruto,
            "neto_tahunan": yearly_neto,
            "pph_tahunan": yearly_pph,
            "pph_bulanan": monthly_pph
        }
        return result
    

# Sample input case 1
# calculator = TaxCalculator()
# result = calculator.calculate_tax(
#     month_worked_list= ['January','February', 'March', 'April', 'May', 'June', 'July', 'August', 'September','October', 'November','December'],
#     tunjangan_hari_raya_dict = {'December': 60_000_000},
#     uang_lembur_dict= {'February': 5_000_000, 'May': 5_000_000},
#     bonus_dict= {'July': 20_000_000},
#     gaji = 10_000_000,
#     tunjangan= 20_000_000,
#     jkk = 0.005,
#     jkm = 0.003,
#     iuran_pensiun= 100_000,
#     iuran_zakat= 200_000,
#     ter_category= "TK/1"
# )

# print(result)

# Sample input case 2
calculator = TaxCalculator()
result = calculator.calculate_tax(
    month_worked_list= ['September','October', 'November','December'],
    #tunjangan_hari_raya_dict = {'December': 60_000_000},
    #uang_lembur_dict= {'February': 5_000_000, 'May': 5_000_000},
    #bonus_dict= {'July': 20_000_000},
    gaji = 15_500_000,
    #tunjangan= 20_000_000,
    #jkk = 0.005,
    #jkm = 0.003,
    iuran_pensiun= 100_000,
    #iuran_zakat= 200_000,
    ter_category= "TK/0"
)

print(result)




            


        

            

            





        




