# 2022-01-11
# Send local image to line notify


"""https://notify-bot.line.me/doc/en/

Parameter name    Required/optional    Type    Description
message    Required    String    1000 characters max
imageThumbnail    Optional    HTTP/HTTPS URL    Maximum size of 240×240px JPEG
imageFullsize    Optional    HTTP/HTTPS URL    Maximum size of 2048×2048px JPEG
imageFile    Optional    File
Upload a image file to the LINE server.
Supported image format is png and jpeg.

If you specified imageThumbnail ,imageFullsize and imageFile, imageFile takes precedence.

There is a limit that you can upload to within one hour.
For more information, please see the section of the API Rate Limit.
"""

#舊版官方範例 https://engineering.linecorp.com/en/blog/using-line-notify-to-send-messages-to-line-from-the-command-line/


#token
token='80mM0Ycf8fx7mxyCaCtemRknn2lBaJI2tndF4bo5WA2'

#api
lme='https://notify-api.line.me/api/notify'

nowtime="Job Done $(date)"

message='Img notify'

curl -X POST -H "Authorization: Bearer ${token}" -F "message=${nowtime}" -F imageFile=@tf.100x100_dark.jpg ${lme}

