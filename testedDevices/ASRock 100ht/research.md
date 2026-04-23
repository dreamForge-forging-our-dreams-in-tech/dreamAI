# Test1

Video: https://youtu.be/b5qUVnDjYc0

Total training time: 8:45
Max temp: 38 degrees
Used ram: ~ 4.44

This was a starting test to collect starting data for upcomming tests.
Interestingly in this test i noticed that it may also respond with the trianing data thats used to help it tell apart the user text message and its own response but this is probally caused by the lack of training data avaialble at this moment.

# test 2

Video: https://youtu.be/__zmbnXrv4Y

training time: 7:30, nearly got 2 minutes of compared to last times training on the ASRock.
used ram: 4.50
max temp: 36 degrees.

Adding all changes of test 8 from the aspire to the ASRock seems to be a significant improvement!, its still not as i would like it to be but its still a major improvement!

Edit: nvm changes turned out to be minor on th ASRock :'(

# Test 4

Video: https://youtu.be/k_vCvxmlRjI

training time: 16 minutes
Used ram: 4.70
Temp: 40 degrees

# test 5 
so i moved test 10 and 11 over to the ASRock and the current loss was around 0.6 and training time around 5:30 which means a cut in time of 30 seconds.

huh this is odd i tested it with 3 out of 4 cores and the training time did not change, interesting, it looks like 75% of the cores can be used but im not sure of that so ill just test this out first :)

# test 6
so i have brought over test 17 to the ASRock and completely forgot that there it has way less cpu cores and so its gonna be way slower, this also caused the qeue of piscina to full up way quicker, causing a crash only after 9 epochs so i have temporary disabled the max qeue number to see what hapens, i can already see some improvements because its a bit faster but it has also been overshooting itself :/

Okay so i also removed the max qeue on my acer aspire 5 and i hit 4:38, holy fuck, im not sure if this will introduce fluctuations again, i really hope not.

Sadly on the ASRock it has become incredibly fast in training but keeps overshooting itself and eventually flat lines between 3 - 4 :/

so the ASRock took 13 minutes to train only for it to keep saying response: response: response:, lol but hey fucking 3 minutes off that is impresive, last training took 16 minutes

ps the second test i ran on my acer aspire 5 took 4:43 seconds and the third 4:50, so it is fluctuating again ###############