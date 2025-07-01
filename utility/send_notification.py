# send notification to slack
import requests

def read_slack_webhook_url(filepath = r"C:\Users\WatabeT\Documents\Git\controlFLIMage\utility\slack_webhook_url.txt"):
    with open(filepath, "r") as file:
        webhook_url = file.read()
    return webhook_url

def send_slack_notification(webhook_url, message = "done"):
    data = {"text": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 200:
        print("Error", response.text)
    else:
        print("sent slack notification")

def send_slack_url_default(message='done'):
    webhook_url = read_slack_webhook_url()
    send_slack_notification(
        webhook_url = webhook_url,
        message = message
    )

def upload_image_to_slack(token, channel, filepath, message):
    with open(filepath, "rb") as file_content:
        response = requests.post(
            "https://slack.com/api/files.upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"channels": channel, "initial_comment": message},
            files={"file": file_content}
        )
        if not response.json().get("ok"):
            print("Error", response.text)
        else:
            print("uploaded image to slack")


if __name__ == "__main__":
    webhook_url = read_slack_webhook_url()
    send_slack_notification(
        webhook_url = webhook_url,
        message = "processing is done"
    )