import config

from MESSAGING_API import twilio,twilio Error

# create a Twython object by passing the necessary secret passwords
twilio = Twython(config.api_key, config.api_secret, config.access_token, config.token_secret)