import smtplib, ssl

me= 'jay.python.development@gmail.com'

port = 465  # For SSL
password = ''#input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

server = smtplib.SMTP_SSL("smtp.gmail.com", port, context=context)
server.login(me, password)


receiver_email = me
message = """\
Subject: Hi there

This message is sent from Python."""
server.sendmail(me, receiver_email, message)


