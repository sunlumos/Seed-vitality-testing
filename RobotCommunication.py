# 用于写入机械手寄存器的代码
import requests

def write_multi_project_d():
    url = "http://10.3.3.29:8080/kndrobotapi/v1/ch0/project_dvar"

    data = [
        {"No": 0, "Value": 1},
        {"No": 1, "Value": 2},
        {"No": 2, "Value": 3},
        {"No": 3, "Value": 4},
        {"No": 4, "Value": 5}
    ]

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code == 200:
        print("Variables written successfully")
    else:
        print(f"Failed to write variables. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    write_multi_project_d()