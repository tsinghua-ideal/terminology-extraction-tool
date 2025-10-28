import pandas as pd
import re



def load_HPCA():
    # List of strings
    result_colums = []

    # default
    hpca_path = "./HPCA" 
    years = [str(year) for year in range(2016, 2026)]
    years.append("MICRO_2020") # Specially, MICRO_2020 in IEEE (csv format)
    for year in years:
        df = pd.read_csv(f"{hpca_path}/{year}.csv")
        result_colums.extend(df["Abstract"].tolist())
    return result_colums



def read_bib(bib_path: str):
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    abstract_pattern = r'abstract\s*=\s*{([^}]*)}'
    
    abstracts = []
    matches = re.findall(abstract_pattern, content, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        abstract = match.strip()
        if abstract:
            abstracts.append(abstract)
    
    # print(f"提取到 {len(abstracts)} 个abstract")
    # print(abstracts[0])
    return abstracts


def load_ACM(conf_name: str):
    # List of strings
    result_colums = []

    # default
    acm_path = f"./{conf_name}" 
    
    if conf_name == "ASPLOS":
        years = [str(year) for year in range(2016, 2023)]
        for year in [2023, 2024, 2025]:
            for i in range(1, 5):
                if year != 2025 or i != 4: # ASPLOS 2025_v4 is not ready
                    years.append(f"{year}_v{i}")
    else:
        years = [str(year) for year in range(2016, 2026)]
        if conf_name == "MICRO":
            years.remove("2020")
            years.remove("2025")

    for year in years:
        result_colums.extend(read_bib(f"{conf_name}/{year}.bib"))
    return result_colums


def load_xlsx(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    return df["英文"].tolist()



if __name__ == "__main__":
    result_HPCA = load_HPCA()
    result_ASPLOS = load_ACM("ASPLOS")
    result_MICRO = load_ACM("MICRO")
    result_ISCA = load_ACM("ISCA")
    

    print(len(result_HPCA))
    # print(result_HPCA[0])
    print(len(result_ASPLOS))
    # print(result_ASPLOS[0])
    print(len(result_MICRO))
    # print(result_MICRO[0])
    print(len(result_ISCA))
    # print(result_ISCA[0])

