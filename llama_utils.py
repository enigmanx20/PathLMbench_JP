import requests

def get_llama_server_models(server_url="http://127.0.0.1:8080"):
    """
    model_name = get_llama_server_models()["data"][0]["id"]
    """
    
    r = requests.get(
        f"{server_url}/v1/models",
        proxies={"http": None, "https": None},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()