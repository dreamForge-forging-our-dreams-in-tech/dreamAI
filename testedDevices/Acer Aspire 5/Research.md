# Test1

Video: https://youtu.be/yWGm-uIacnM

Total training time: 2:40
Max temp: 38 degrees
Used ram: ~ 7

This was a starting test to collect starting data for upcomming tests.

# test 2 with Evie

total training time: 2:20

max temp: 41 degrees
used ram: ~6

Keeps stuck at a loss 0f 3.3673

changed learning rate from 0.01 to 0.001 or 0.1 > got incredibly slow at learning and had fluctuating results.

Okay turns out grad is always undefined in Evie.js which causes the empty tensor detecting to skip the loop.

Okay turns out model.fit was returning an array and not a object, took some time to figurie it out with gemini.
, if gemini just fucking told me immediatly i needed to acces tensor in the object instead of a fucking number.

# test 3 evie looping array and properly accesing objects and tensors

total training time: 2:45
max temp: 38 degrees.
used ram: ~6

Reposne: similar results as to adam.

# teest 4 evie with batching

Slower, training takes longer and loss has become worse than before
So back to the drawing board.

# test 5 with tidy and removed dataSync

Slightly improved training speeds it got under 0.5 after 22 (25 previously) epochs in 1:50 minutes
Above results do seem to vary

# test 6 imrpoved math, shorter is faster

this:
                const nextM = m.mul(this.beta1).add(grad.mul(1 - this.beta1));
                const nextV = v.mul(this.beta2).add(grad.square().mul(1 - this.beta2));
Became:
                const nextM = m.add(grad.sub(m).mul(1 - this.beta1));
                const nextV = v.add(grad.square().sub(v).mul(1 - this.beta2));

Minor improvements :(

# test 7

this: const update = mHat.mul(this.learningRate).div(vHat.sqrt().add(this.epsilon));
to: const update = mHat.mul(this.learningRate).div(vHat.add(this.epsilon).sqrt());

This provided a small but major improvement, it now would reach under 0.5 before 25 instead of around 25(23 - 27) like previous improvements did.
Because of this it seems responses also improved but due to limited training data this is up to doubt

Test 7 - pre computing

From:

                const nextM = m.add(grad.sub(m).mul(1 - this.beta1));
                const nextV = v.add(grad.square().sub(v).mul(1 - this.beta2));

                const mHat = nextM.div(1 - Math.pow(this.beta1, this.t));
                const vHat = nextV.div(1 - Math.pow(this.beta2, this.t));
To:
            const biasCorrection1 = 1 - Math.pow(this.beta1, this.t);
            const biasCorrection2 = 1 - Math.pow(this.beta2, this.t);

            const nextM_math = 1 - this.beta1;
            const nextV_math = 1 - this.beta2;
            
                const nextM = m.add(grad.sub(m).mul(nextM_math));
                const nextV = v.add(grad.square().sub(v).mul(nextV_mathz));

                const mHat = nextM.div(biasCorrection1);
                const vHat = nextV.div(biasCorrection2);

This added minor improvements, its loss still gets under 0.5 around 22 - 24 epochs but the number is closer/lower than previously (e.g. 0.499 to 0.480)
Sometimes it hits 0.3 now before 22.

# test 8 - video

video: https://youtu.be/yiyyAPtIuXk

Training time: 2:30
max temp: 37 degrees
used ram: 6.30

-- added scrolling to bottom if the users messages overflows.

-- shortened math formulas for increased speed, although it did not give much improvement.

-- added pre computing to Evie its for loop.

# test 9

Video: https://youtu.be/GtW99mZPhRA

Training time: ~ 5 minutes (with extra training data)
Used ram: 6.30
temp: 37;

So i was thinking for a bit and thought of adding a tokenizer to the training data, this seemed to be a really good idea as this significantly cut training time (40 seconds) but it has come with a cost of the loss number going under 0.5 at 25 again :(

Okay i added the de_tokenizer and fixed a array parsing issue and training times cut to under 30s but the resposnes turned into: hellodreamAIe+your0000hellodreamAIe+your0000hellodreamAIe+your0000hellodreamAIe+your0000hellodreamAIe+

Still masssive improvemenets in training which is really great.

so this happens becuase i was parsing tokenized strings with a sceintific notation which the AI could process so those notations where now turned into arrays and that solved the issue, which made things far better but slowered the training time obviously but its still a 60 second cut from revious tests!
Loss starts now at 2.5 instead of 3.6!

The Ai still says random bs tho: ",is,is,my,?, ,." but that is most likely because i forgot to tokenize the users input.

Added user tokenizing input and the AI keeps saying random bs: ,is,is,my,?, ,.
So i grabbed gemini and asked whats causing it, turns out tf needs different layers for numbers or words (i keep learning new things!)

Anyways after a lot of experimenting i managed to ge the ai to actually make sentences this time albeit it, it loops and spits out the training data instead of deciding itself, but its done training under 60 seconds so thats really great!

I fixed the above but now it just spits out random stuff so ill have to dig into the training data some more.

More training data only caused it to take longer and spit out more random bs, now testing with a higher epoch count of 120

Okay ehm gemini helped me a bit with my code, i swear i have to disable gemini if i want to learn how a llm works!
Anyways we changed its generate loop so it gets the best possible result because otherwise itll just go soup mode.
interestingly this added changes to the speed to, it went from 30 minutes to 5, interesting.

Sending the message to the AI just like how it has been trained (context: . user: what is your name?. response:) has imrpoved the responses and it now actually returns coherent sentences!
Sadly due to lack of data it can still throw out incorrect responses :(

Using decimal tekenization improved training speed but the AI keeps returning 0 0 0 0 0 0 0 0 0 0 0 0 0 :(
It was a good idea but since tensorflow already turns numbers into decimals it gets the math confused so i'd be better of optimising either way tensorflow to work better with decimal tokenization or optimise all code to make it go faster.
The improved speeds where probally because it just keeps spitting out zeros so it was probally jsut training on that padding (which is 0)
Im curious now if the padding of 0 is actually what messses up the decimal tokenization because it kept spitting out arrays like so [[0.0], [0.0], [0.0], [0.001]] which then could be why the AI spits out so many zeros but this also happens with full numbers the only differnece is its not a decimal anymore, hmm interesting.
Did a quick test and it doesn't seem like that that is the issue :(

Decimals and integers will result in the same speeds if i where to be able to make it work, 1 + 1 and 0.1 + 0.1 take almost the same amount of time to calculate.

# test 10

I was thinking of adding multiple cpu core threading and first added this to the loss function/applyGradients, this seemed to be a awesome idea as this massively improved training quality albeit that it only cut training time with ~ 5-20 seconds.
Weird thing is tho that even tho the loss drops quicker its responses are a bit well good quality or not so nothing much changed ...

I have tried splitting the training to all available cores aswell however due to the complexity i used Gemini to help me which resulted in the picture in this folder named test10_Core_training.png

Edit: I'm going back to the drawing board with this one
# test 11

Added optimiser worker threads to all available cpu cores instead of a default number of 4.
Interestingly using more cpu cores on the optimiser has increased training (varying, somethimes its faster) time and but the loss number did reach 0.5 quicker.

I also tried out loading the workers of the Evie optimiser when the server starts however this caused the training to flatline at 4.9 because the workers need to be tied to the class and not globally to avoid confusion.

interestingly if i initialize the evie optimiser in a global variable it does work.

I made os.availableParallelism() into os.availableParallelism() - 2 to ensure that not all cores are used by the workers, so the main thread has some breathing room, interestingly this worsened the loss number and training time.

Interestingly i now changed it into os.availableParallelism() / 2 which seemed to have improved the quiality and decreased training time, albeit the training time got decreased from above statements as we already hit the 5 minutes mark, this means that technically there isnt much improvement in time.

Well al the above seems to be very varying, ugh.

I made the Evie optimiser initialize on compilation again and that improved time on the first test but if its gonna be anything like the above, it will probally be worser the second test.

Just like i expected to happen it was slower and worser the second time.

I went back to using all cores again and surprise surprise it didnt matter, nothing improved.

Huh i went to using the optimiser on 2 cores and it hit 4:54 m:ss:mm however the loss number is less close to 0.5, interestingly, i dont think that this will do much on the ASRock because it only has 4 cores but its worth a test atleast.

Its also varying again, FRICKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK, test 2 with this was 5 minutes instead of 4:54, test 3 was 4:54 (think this change can stay even if its small, every improvement is welcome!), test 4 was 4:49, interesting this actually improved training time!

Now testing it with 3 cores! but tbh im rather curious how this will be on my ASRock because that has 4 cores instead of 10, 3 cores is 30 seconds slower?????? im so confused by this tbh.

I went back to 2 cores again and this test was around 5 minutes and had a loss of 0.57 so the quality was better but not the time, interesting, it seems to be either have a better time or a lower loss number, did another test and it was around 5:10 wth is going on here.

WTF is happening it just had a training time of 5:30 ????????, i ran another test with my device idle and it went to 5:01.

WTH is going on it just went to 5:13 why does it vary so fucking much??
Did another test ended up at 4:59, im so fucking confused as to whats going on here, why does it vary so much?

so i did a quick research with gemini and he said that the v* engine needs to warm up and that the cpu core processes are moved around and suggested the first 2 options/task last one is mine just to see what happens with that.

* use a new command to lock the used cpu cores to prevent process from being moved around - check!
* test warming up of V8 engine - check
* check how everything is influenced by more variable core use

# test 12 
added a simple code to start using 75% procent of available cpu cores, to see how this improves time and more importantly how this improves with a warm up of the V8 engine.

first test was 5:12 with a loss of 0.58, second one 4:59 with a loss of 0.59, test 3: 4:57 with a loss of 0.59, test 4: 4:58 with a loss of 0.65, test 5: 5:14 with a loss of 0.56, test 6: 5:26 with a loss of 0.64, this one is presumably higher because i was doing more tasks on the cpu.

okay i switched to starting the node process with: taskset -c 0-8 node server.js

this improved results a little bit, it stays under five minute slonger and even went under 4:54 twice! howver just like with the regular node server.js it eventually starts taking longer again.

bottom line:
* it needs a warm up
* process arent allowed to be moved around.
* it needs to have something to prevent memory leaks or atleast prevent the memory buss from getting filled too much, because thats when its starts taking longer to train, could possibly be fixed by having a dedicated training script that stops running once training gets finished.

test to use full 100% of the available cpu cores when 75% is only opened to the process.

Okay so i went on and tested how using 100% of the 75% avaialble cores runs and well, its crap still same effect as previously only drawback is its always above 5 minutes, so i think using 75% of 75% is great, ill test using all cores with taskset tomorrow.

so i switched to: taskset -c 0-10 node server.js

test 1: 4:53 with a loss of 0.62
test 2: 4:52 with a loss of 0.60
test 3 4:53 with a loss of 0.59
test 4: 4:54 with a loss of 0.59
test 5: 4:56 with a loss of 0.59

using all cores while leaving 2 cores available for the os and preserving one core for the main thread has seemed to be a very good choice as we now have a traing time of around 4:50 minutes, warming up will only cut of 1 second of the training time so its not nessecary (how do i spell this word again) unsurprisingly after it reaches test 5 it starts taking longer again so im very curious as to why that is happening, ill have to run another 5 test to see if it has something to do with memory orsm.

okay so i just did the test and its well dissapointing

test 1: 4:56 with a loss of 0.6
test 2: 5:04 with a loss of 0.63
test 3: 5:06 with a loss of 0.59
test 4: 5:07 with a loss of 0.61

this is well sad but its good to see it started under 5 minutes tho!.
well i jsut ran another test round and this time it had the same results as the first test.

im keeping this change because it added improvements albeit weird ones so id have to look into memory management.

I added propper tokenization for spaces aswel and this caused training tiem to go under 4:50 but due to impropper memory management it still goes up with 2 seconds every time.

i just ran a quick test and it seems that there is no memory leak, so i wonder what exactly is going on here, something is slowing things down but what?

There is really something weird going on i jsut trained it again and it took 5:10, if its no memory leak then what is it?

i just did another test and it took 4:44 minutes to train, what the hell is going on here.

# test 13
I installed piscina because it should make the handling of worker threads easier, sadly the first test stayed around 5:15 which isnt a imporvement but actually worse, it still is variable and keeps getting longer with every test i ran.

I decided to go to manual clean up of tenser and this made the training time shorter and now it only gets lojnger by 1 second instead of 2 but it is still taking too long.

i added manual garbage colelcting and this causes too much overhead, it started great at 5:04 but increased with 3 seconds every test.