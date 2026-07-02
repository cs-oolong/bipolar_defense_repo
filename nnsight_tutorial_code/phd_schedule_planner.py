dias_da_semana = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
horarios = ["manhã", "tarde", "noite"]

tuplas = []
for possivel_ida in dias_da_semana:
    for possivel_horario_ida in horarios:
        for possivel_volta in dias_da_semana:
            for possivel_horario_volta in horarios:
                tupla = (
                    possivel_ida,
                    possivel_horario_ida,
                    possivel_volta,
                    possivel_horario_volta,
                )
                tuplas.append(tupla)

print(len(tuplas))
print(tuplas)

import csv

caminho_csv = "phd_schedule_planner.csv"

with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(["dia_ida", "horario_ida", "dia_volta", "horario_volta"])
    escritor.writerows(tuplas)

print(f"CSV salvo em: {caminho_csv}")
