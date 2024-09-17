import requests
import fire


def _make_request(prompt: str):

    url = "http://127.0.0.1:8080/generate"

    data = {
        "prompts": [prompt],
        "max_length": 50
    }

    response = requests.post(url, json=data)
    result = None

    if response.status_code == 200:
        result = response.json()
        print("Generated Text:", result['completions'])
    else:
        print(f"Error: {response.status_code}, {response.text}")

    return result

def main():

    res = _make_request("Once upon a time",)
    print(res)

if __name__ == "__main__":
    fire.Fire(main)