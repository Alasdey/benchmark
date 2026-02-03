import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email details
from_email = "no-one@yours.fr"
to_email = "lenalasdey@gmail.com"
subject = "Test Email from Alias"
body = "Hello, this is a test email sent from an alias without revealing the original address."

# SMTP server config (your domain's server)
smtp_server = "smtp.yours.fr"
smtp_port = 587
smtp_user = "no-one@yours.fr"
smtp_password = "your-password"

# Create the email
msg = MIMEMultipart()
msg['From'] = from_email
msg['To'] = to_email
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

# Send the email
with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.send_message(msg)

print("Email sent successfully.")