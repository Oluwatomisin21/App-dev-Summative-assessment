# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'ACc3a3ea8f5a93fbccd1512df3330b408e'
auth_token = 'd4d5740eb74e1aaad7347285cfc2e785'
client = Client(account_sid, auth_token)

message = client.messages.create(
                              from_='whatsapp:+2349068930947',
                              body='there is a forecasted combined total of less than 4 MW,please take preventive actions to avoid no-compliance on SLA',
                              to='whatsapp:+2349062796251'
                          )

print(message.sid)