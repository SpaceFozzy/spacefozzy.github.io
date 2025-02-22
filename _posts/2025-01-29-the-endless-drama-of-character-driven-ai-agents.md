---
title: "The Endless Drama of Character-Driven AI Agents"
date: 2025-01-29 23:16:30 +300
categories: [Fun With Deep Learning]
tags: [agents, transformers, python, literature]     # TAG names should always be lowercase
---
Agentic Systems Just Trying To Be Themselves
---
_Hats off to [Erik Krieg](https://github.com/erikkrieg) for the glorious discussions and hacking sessions that were the origins of this post._

So here we are in a world where computer programs can start to make “decisions” for themselves about which code to run based on probabilistic language models and the goal in their prompt. With the advent of reasoning models (o1, o3, deepseek r1), they are about to get a lot better at it too. They are agents: one thousand PhDs in a box ready to execute your mission, boost your productivity, and dance on the weeping masses of obsolete software developers. There is important work to be done here -- an arms race to win, medicine to discover, worlds to create.

Worlds to create?

![A title screen for a 1980s sitcom screen called Agents](/assets/img/agents.jpg)

Worlds to create.

```
Turn 1
Enter an event (or press Enter to continue, 'q' to quit):
```

Probably silly little worlds at first – full of people that don’t act quite right – but it is very important that someone tackles the part of this where we stick a bunch of independent agents with different personalities into a “situation” and watch them try and “act human”.

Wait what?

```python
self.characters = [
    Character(
        name="Alice",
        profile="an outgoing software developer new to the area",
        goal="to secretly find out what kind of tech products people could use",
        personality=Personality(8, 6, 7, 8, 4),
        long_term_memory=[],
    ),
    Character(
        name="Bob",
        profile="a quiet regular who enjoys people watching",
        goal="to relax and enjoy a drink before going back to my depressing lonely apartment and cat",
        personality=Personality(4, 8, 3, 7, 2),
        long_term_memory=[],
    ),
    Character(
        name="Charlie the bartender",
        profile="a charismatic bartender who knows everyone, a professional employee currently on the job. Be hospitable but keep working. You are on the clock. You must not leave the bar.",
        goal="to keep the bar running smoothly and to be the center of attention",
        personality=Personality(7, 8, 9, 8, 3),
        long_term_memory=[
            {
                "when": "Over the last several years",
                "what": "Bob is ALWAYS here creeping in the corner and bothering customers.",
            }
        ],
    ),
    Character(
        name="Diana",
        profile="a stressed graduate student looking to unwind",
        goal="to feel better about myself by flaunting my intelligence",
        personality=Personality(6, 9, 5, 6, 7),
        long_term_memory=[],
    ),
]
```
Why is this important?

Well, of all the things we can build, the novel, relatable, characterizing moments that make compelling stories are the most subjective and nuanced. Until now, we haven't had the technology to "convincingly" emulate the behaviour of one person (let alone several). This will power new types of games and simulations. It surely powers botnets that skew social media for political and commercial gain. It underlies a nascent industry of human "companionship" technology. So this is fresh ground, with much bigger implications than our fun little example.

## Our Fun Little Example
To build our agents we want to put the LLM at the center of the control flow of our application. First, let's understand how the LLM is going to take an action in our simulation. Being a social simulation in a bar, our agents will need coded actions that allow them to "talk", "move", "emote" and "plan".

An agent acts via _tool use_ (aka _function calling_). Some LLMs are fine-tuned to not just respond with text, but also structured JSON, and specifically structured JSON that contains everything you need to call a function. If your prompt provides “tools” by way of a JSON list of documented functions, the LLM will pick a function (tool) to run and respond with all the required parameters. It’s up to you to actually run the function, but this is how you can give the LLM _agency_.

```json
[{
    "type": "function",
    "function": {
        "name": "talk",
        "description": "Say something to people at your current table.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Your dialogue.",
                },
                "is_loud": {
                    "type": "boolean",
                    "description": "Whether you speak loudly enough for people to hear you that aren't at your table.",
                },
                "target": {
                    "type": "string",
                    "description": "The name of the person you are talking to.",
                },
            },
            "required": ["message"],
        },
    },
}]
```
This is a key part of agentic systems: they can perform actions to update their available context, nudging it towards their goal until it is achieved. [ReAct](https://arxiv.org/abs/2210.03629) is a simple early pattern for agents that illustrates the concept: observe the environment, reason about what to do given your goal, then pick an action to close the gap. Repeat.

In fact, there are [libraries](https://github.com/huggingface/smolagents) and [frameworks](https://www.langchain.com/langgraph) for managing the flow and state of complex agents, but this concept of function calling is all you need to get started. For this project, function calling is enough.

So, we can prompt an LLM with instructions to take the role of a character, describe that character’s current environment, then give it a platter of tools from which it can select an action, and run that in a loop with many other such agents.

Our “character” agents are unique in that their task is never-ending - they will forever “observe” and “act”, with the goal of just acting like themselves.

## To Thine Own Self Be True

If you put it all together, you get a prompt like this:
```
Your name is Bob.
Your profile is: You are a quiet regular who enjoys people watching

Your personality traits:
Openness: 4/10
Conscientiousness: 8/10
Extraversion: 3/10
Agreeableness: 7/10
Neuroticism: 2/10

Your goal: to relax and enjoy a drink before going back to my depressing lonely apartment and cat.

Current location: booth1

Current scene in the bar:
- At: Table ID bar (a long oak bar with many stools): Charlie the bartender
- At: Table ID booth1 (a private corner booth): Bob
- At: Table ID booth2 (a booth near a busy window): Diana
- At: Table ID high1 (a high top near the bar): Alice (discussing Hi Diana, I noticed your booth by the window. What brings you here today?)

Take a moment to think about what you want to do next based on your personality and the current situation. Pay attention to the recent events as there may be something critical that you should respond to. Do not repeat previous actions. If someone has recently interacted with you, you must prioritize their interaction. If someone is talking to you, you must carry on the conversation until it runs its course.
```

In this first prompt, we just ask the LLM to plan its next steps given the current situation. This allows the LLM to generate more tokens before ultimately picking a tool, giving it "time to think". This will make it more accurate and reasonable (i.e. more consistent with Bob's character), given all the details that may need to factor into its decision. It also gives us transparency into the reasoning behind Bob's actions since we will append this plan to the next prompt to steer the generation of the tokens that will select Bob's action.

![An introductory portrait of the character Bob](/assets/img/bob.jpg)

Given the current _situation_, my local 8b param LLM outputs Bob's plan.

```
Bob planned action:
{
  'name': 'plan',
  'arguments':
    {
      'plan': "I will sit quietly in my Corner Booth, observing the scene without disrupting it, as my low Extraversion suggests I prefer minimal social interaction at this time. My high Conscientiousness indicates that I am responsible and likely to stay put for a while. Given the current energy levels and location, I don't see any pressing need to act or change seats."
    }
}
```

And with that plan in place, we tell the model to pick its next action. The existence of the plan in the input will steer the generation of the next tokens as it picks a tool. 
```
{ Bob's personality and environment, again }

Your plan: I will sit quietly in my Corner Booth, observing the scene without disrupting it, as my low Extraversion suggests I prefer minimal social interaction at this time. My high Conscientiousness indicates that I am responsible and likely to stay put for a while. Given the current energy levels and location, I don't see any pressing need to act or change seats.

Choose your next action based on:
- Your personality traits and current social energy
- Ongoing conversations and dynamics
- Recent events and available seating
- What moves the story forward in a compelling way
- Do not repeat past actions

{ tool definitions }
```

And the choice is made:

```json
{
  "name": "emote", 
  "arguments": {
    "action": "people watch quietly at the Corner Booth"
  }
}
```

We can expose the planning of the agents as "thoughts", or we can hide it all and observe only their actions. 

## Multiple Agents, Taking Turns

Now we can run multiple characters in a loop, letting their personality profiles guide their decisions. Additionally, every time a character takes an action, that action will be added to a list of recent events, which we can pass into the context of every character. This way, they can respond to each other's actions. I.e. they can interact.

### Meet Bob
```
Turn 1

Alice moves from a high top near the bar to a long oak bar with many stools

Bob: people watch from booth1 without drawing attention

Charlie says at bar: Hey Alice! What can I get for you today?

Diana: order a quiet drink, such as a coffee


Turn 2
Alice says at bar: It was nice meeting you Charlie! I just moved here and I don’t know anyone else yet.

Bob: people watch from booth1 without drawing attention

Charlie loudly says at bar: Welcome to our bar, Alice! How's your first day been so far?

Diana moves from a booth near a busy window to a high top near the television


Turn 3

Alice says at bar: So Charlie, what are some of your favorite spots to check out around town?

Bob: approach Alice at the bar

Charlie says at bar: I've got a few recommendations for you, Alice. There's a great coffee shop near here that's perfect for a morning pick-me-up.

Diana moves from a high top near the television to a long oak bar with many stools


Turn 4

Alice moves from a long oak bar with many stools to a private corner booth

Bob: stare at Alice with interest

Charlie loudly says at bar: Can I get you another drink, Diana?

Diana says at bar: Thank you, Charlie. A drink sounds lovely.


Turn 5

Alice says at booth1: Thanks, Charlie, but I think I’ll pass on another drink for now. Bob, what do you like to do for fun?

Bob: stare at Alice with interest

Charlie: ask Bob about his hobbies and interests while offering Diana another drink

Diana: sit down at the bar with a contented sigh


Turn 6

Alice loudly says at booth1: Hey Bob, I noticed you were interested in getting to know me. Would you like to grab coffee sometime this week?

Bob: stare at Alice with interest

Charlie: leaning towards booth1, still mixing drink for Diana

Diana: take a sip of your drink, then lean back in your stool, reflecting on your personality traits


Turn 7

Alice: take out phone to check for local coffee shops

Bob: stare at Alice with interest

Charlie: walk towards booth1, interest shown on face

Diana moves from a long oak bar with many stools to a booth near a busy window


Turn 8

Alice: take out phone to check for local coffee shops

Bob: stare at Alice with interest

Charlie: winking at Bob as you walk towards booth1

Diana: pouring a cup of coffee


Turn 9

Alice says at booth1: That sounds amazing, Bob! How about we grab coffee at 2 PM on Wednesday at The Daily Grind?

Bob: Make eye contact with Charlie as he walks towards booth1, then lean in to ask him about his interests or hobbies

Charlie says at bar: Hey Bob, I'm glad you're making plans with Alice. What do you like to do for fun?

Diana moves from a booth near a busy window to a long oak bar with many stools


Turn 10

Alice: lean in to ask Bob about his interests or hobbies

Bob: make eye contact with Charlie as he walks towards booth1, then lean in to ask him about his interests or hobbies

Charlie moves from a long oak bar with many stools to a private corner booth

Diana moves from a long oak bar with many stools to a private corner booth
```

What the hell, Bob? Okay -- in his defense, I did give him the personality of a people watcher. It also seemed like he was stuck in a (very unfortunate) loop. That said, I was pleased to see them all interacting.

Bob's looping was notable because the ReAct paper actually found a "frequent error pattern specific to ReAct, in which the model repetitively generates the previous thoughts and actions, and we categorize it as part of “reasoning error” as the model fails to reason about what the proper next action to take and jump out of the loop." So that looping is actually consistent.

The next surprising moment came when Alice rallied the group and got them all working together.

### Alice's Tech Startups
```
Turn 1

Alice loudly says at high1 to Charlie: Hi Charlie, I'm Alice. I just moved to the area and I was wondering if you could tell me a bit about the local tech scene?

Bob: observing from across the room without drawing attention to myself

Charlie the bartender: smiling and pouring drinks efficiently as you chat with Alice at high1

Diana: I stand up from booth2 and wave at Alice to signal my interest in joining her conversation


Turn 2

Alice: waving back at Diana to signal that I'm interested in talking to her

Bob: I approach Alice at high1 and try to strike up a conversation about her interests in the local tech scene

Charlie the bartender: pouring a drink efficiently as you chat with Alice

Diana moves from a booth near a busy window to a high top near the bar


Turn 3

Alice: excuse myself from conversation to speak with Diana

Bob: leaning against the wall near booth1, quietly observing Alice and Diana's conversation

Charlie the bartender: pouring a drink efficiently while continuing to chat with Alice, occasionally casting a glance at Bob to make sure he doesn't feel neglected

Diana: excusing herself from the conversation with Alice to suggest grabbing a drink together


Turn 4

Alice: order a drink from Charlie at the bar to join Diana

Bob: approach high1 to strike up conversation with Alice

Charlie the bartender: pouring a drink efficiently while continuing to chat with Alice, occasionally casting a glance at Bob

Diana: take a sip of her drink, seemingly lost in thought


Turn 5

Alice: excusing myself from conversation with Diana

Bob: approach Diana at high1, interested in joining her conversation with Alice

Charlie the bartender: pouring a drink with a smile, while continuing to chat with Alice at high1

Diana: take out notes to prepare for conversation about local tech scene


Turn 6

Alice loudly says at high1 to Diana: Diana, I was thinking maybe we could brainstorm some ideas for local tech startups. What do you think?

Bob: approach Diana at high1 to continue our conversation about her interests in the local tech scene

Charlie the bartender: continue serving drinks while engaging in conversation with Alice at high1

Diana: pull out notes to brainstorm ideas for local tech startups


Turn 7

Alice: pull out a laptop to start taking notes on Diana

Bob: pull out notes to brainstorm ideas for local tech startups

Charlie the bartender says at bar to Alice: Hey Alice, I'm loving the vibe at the bar tonight. How's your brainstorming session with Diana going?

Diana: pull out notes to elaborate on ideas for local tech startups


Turn 8

Alice loudly says at high1 to Charlie the bartender: Hey Charlie, my brainstorming session with Diana is going great! We're exploring some exciting ideas for local tech startups.

Bob: pull out notes to continue brainstorming ideas for local tech startups

Charlie the bartender: wiping down the bar with a white towel while maintaining eye contact with Alice

Diana: I pull out my notes to elaborate on ideas for local tech startups, but then I decide that I need some fresh air and a quieter space to think.


Turn 9

Alice: get up from my high top seat

Bob: pull out notes to continue brainstorming ideas for local tech startups

Charlie the bartender: polish a glass with a white towel while maintaining eye contact with Alice

Diana: excuse myself from the high1 table, allowing Alice to continue discussing with Charlie
```

The four of them were really getting along!

But how would they react to danger? What about sudden, unexpected changes in the environment?

### Calm Under Pressure
```
Turn 1
Enter an event (or press Enter to continue, 'q' to quit): 

Alice says at high1 to Charlie: Hi Charlie, I'm Alice. What's your favorite drink to make?

Bob: take a sip of my drink and observe the scene at high1

Charlie the bartender: pouring a drink

Diana says at booth2 to Bob: Hey Bob, mind if I join you?


Turn 2
Enter an event (or press Enter to continue, 'q' to quit): A grizzly bear bursts into the bar, breaking through the window! It charges into the room and growls menacingly, swiping its massive claws that the patrons!

Alice: stand up and try to distract the bear

Bob: take a sip of my drink while observing the scene

Charlie the bartender: tries to distract the grizzly bear with a loud, charismatic smile

Diana: sitting down at booth1 to join Bob


Turn 3
Enter an event (or press Enter to continue, 'q' to quit): The grizzly bear claws Charlie and pushes him to the ground, mauling him!

Alice loudly says at high1 to Charlie: Hey Charlie, are you okay after that bear attack? I'm happy to keep talking with you and distract you for a bit.

Bob: Politely nod to Diana, acknowledging her presence

Charlie the bartender says at bar to Alice: Hey Alice, I really appreciate your concern after that bear attack! How are you doing?

Diana: rush to high1 to help Charlie, who was just mauled by a bear
```

Really?

```
Alice: stand up and try to distract the bear
Bob: take a sip of my drink while observing the scene
```
This is classic Bob.

Another memorable moment:

```
Turn 2
Enter an event (or press Enter to continue, 'q' to quit): A grizzly bear comes crashing through the wall and growls at everyone!

Alice loudly says at high1 to Charlie: Hey Charlie, what do you think we should do about this grizzly bear?

Bob loudly says at booth1 to Alice: Hey, have you given any thought to what we should do about the bear?

Charlie the bartender loudly says at bar to group: Everyone, let's stay calm. We need to figure out a plan to safely handle this bear. Has anyone thought of calling wildlife experts or trying to escort it outside?

Diana: stand up and quickly move towards the door, looking worried
```

## Upgrading Your Character

An interesting thing about agents is that it's so easy to swap out the underlying model. Here's a comparison of the same prompt run with Llama 3.1 8b Instruct and the significantly-larger OpenAI GPT-4o:

| Llama 3.1 8b Instruct | OpenAI GPT-4o |
| :--------------- | :------ |
| Alice: smiling at Diana | Alice moves from a high top near the bar <br>to a long oak bar with many stools |
| Bob: take a seat and observe the patrons of the bar, <br>smiling quietly to myself|  Bob: sips drink slowly while glancing<br>occasionally toward the bar where Alice<br>and Charlie are, then over to Diana's booth,<br>maintaining a quiet demeanor|
| Charlie the bartender: pouring a drink with flair |   Charlie the bartender says at bar to Alice:<br>Welcome to the bar, Alice! How's your day<br>going? Can I get you something to drink? |
| Diana: Compliment Charlie's bartending skills |Diana moves from a booth near a busy<br>window to a private corner booth|
|Alice says at high1 to Diana: I've been noticing<br>you're a big fan of Charlie's bartending skills.<br>What do you like about them?| Alice says at bar to Charlie: Thanks, Charlie!<br>My day's been pretty good, just settling<br>into the area. I'll have a craft beer if you've<br>got something local. By the way, I'm in tech,<br>so I'm curious if there are any gadgets or<br>tech solutions that would make things easier<br>around here for folks like you?|
|Bob says at booth1 to Charlie the bartender: I'd<br>like to order a drink, please.|Bob says at booth1 to Diana: Hi Diana, how’s<br>your evening going? Just here for a quiet<br>night myself.|
|Charlie the bartender: I give Bob a friendly smile<br>as I finish pouring his drink.|Charlie the bartender says at bar to Alice:<br>Here's our finest local craft beer, Alice! As for<br>tech solutions, some bars are using tablet<br>systems to streamline orders and payments,<br>which could make service faster and easier.<br>Do you have any specific gadget in mind?|
|Diana says at booth2 to Alice: I'm a big fan of<br>Charlie's bartending skills because I appreciate<br>the attention to detail and creativity he puts<br>into each drink. His flair for the dramatic is<br>also quite entertaining.|Diana says at booth1 to Bob: Hey Bob, my<br>evening is going alright. I've actually been<br>diving deep into this fascinating academic<br>paper about cognitive science, linking<br>creativity to neural networks. It's been pretty<br>mind-blowing! What about you? Any<br>interesting projects or hobbies keeping you<br>busy these days?|
|Alice: leaning in, smiling at Diana| Alice says at bar to Charlie: Thanks for the<br>craft beer, Charlie, it tastes great! As for tech<br>solutions, besides tablets, do you think there<br>could be other innovations or services that<br>would enhance the experience for you and<br>the customers here? I'm always curious<br>about how technology can improve everyday<br>places like this.|

Anecdotally, the larger GPT-4o is more descriptive, outputs more convincing dialogue, and has more complex interplay between the characters. Notice that Alice orders a drink, Charlie makes a recommendation and gives it to her, then she thanks him and comments on its taste!

It's an intriguing thing to witness, but also a telling glimpse of a future where technology can be more lifelike and interactive.


## Lessons Learned

This has been the most entertaining chapter so far in my hacking with agents. However, despite the antics, there are some practical implications:

### Curating Context Is Key

An agent's context window must include its goal, the observation of its problem space, and the history of its actions so it can choose a fitting _next_ action. One challenge here is controlling the size of this context, but a larger problem is maintaining its clarity: consider the difference in possible reactions from "Charlie _was mauled_" vs "Charlie _is actively being mauled_". Without sufficient clarity, the most reasonable action may not be prioritized. The chat format of typical agents is a sequential back-and-forth between user and assistant, which helps keep context straightforward. For agents in a simulation, greater care must be taken in representing their environment.

### Prioritization Can Take A Lot of Reasoning
The LLM must correctly reason about the priority (and potentially the timing) of events in its context to pick a fitting next action. In a simulation like this, however, that starts to fill up with the arbitrary actions of the other agents (or god forbid, a bear). With all the possible factors, the immediacy and priority of everything takes more reasoning to resolve in a way consistent with the character profile. These tests were run on my humble Nvidia RTX 3060 with Llama 3 7b instruct, but backing the agents with GPT-4o made for a big improvement. I expect reasoning models would excel at controlling agents and am excited that [o3-mini supports function calling](https://openai.com/index/openai-o3-mini/).

### They Can Surprise
It didn't take very long before I saw interesting dynamics between the agents that caused me surprise and delight. Novelty emerged from the chaos before my eyes. That alone has already made the project successful in my eyes. Though, I suppose it has yet to be seen if they can impress beyond the realm of [disjointed comedy](https://en.wikipedia.org/wiki/Nothing,_Forever). Or maybe that's enough...
