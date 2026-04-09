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