# Dik
# asumsi kerja setahun

# TER A
ter_cat = "k/0"
ter_0 = "58.5jt"
ter_persen = "13%"


## Asumsi sama perbulan (no bonus)
Gaji = "10jt"
Tunjangan = "20jt"
jkk = "0.5%"
jkm = "0.3%"
natura = "1jt"
iuran_pensiun = "100k"
iuran_zakat = "200k"

## Tiap bulan beda (bonus berubah-ubah)
Gaji = "10jt"
bonus = "thr + lembur + bonus"
jkk = "0.5%"
jkm = "0.3%"
natura = "1jt"

# bruto bulanan. This value is used to fetch ter percentage
bruto_bulanan = Gaji + Tunjangan + ((jkk+jkm)*Gaji) + natura

# pph21/bulan (except december)
pph_bulanan = bruto_bulanan * ter_persen

# pph21 ampe bulan 11
pph_ampe_bulan_11 = "asumsi namb dari bulan satu ampe 11"

# bruto tahunan
bruto_tahunan = "nambah ampe bulan terakhir dari bulan pertama"

# neto tahunan
biaya_jabatan = "5%" * Gaji * 12  # cannot be more than 500k/bulan
iuran_pensiun = "100k"* 12
iuran_zakat = "200k" * 12

neto_tahunan = bruto_tahunan - biaya_jabatan - iuran_pensiun - iuran_zakat

# ptkp
ptkp = neto_tahunan - ter_0

# pph tahunan
pph_tahunan = "curi punya ibu bos"

# bulan december
pph_bulan_12 = pph_tahunan - pph_ampe_bulan_11

# Kunci jawaban










