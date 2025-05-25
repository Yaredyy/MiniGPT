import random

user_questions = [
    "Hi!",
    "How are you?",
    "What's your name?",
    "Tell me a joke.",
    "What's the weather like?",
    "What's 2 + 2?",
    "Can you help me with homework?",
    "What do you think about AI?",
    "Who made you?",
    "What's your favorite food?",
    "Tell me something cool.",
    "I'm sad.",
    "What should I do today?",
    "Can you sing?",
    "How old are you?"
]

bot_responses = [
    "Hey there!",
    "I'm feeling digital but delightful!",
    "I'm just code, but you can call me ByteBuddy.",
    "Why did the robot get promoted? Because it was outstanding in its field",
    "I can't see weather, but I bet it's cloudy with a chance of memes.",
    "2 + 2 = 4. Easy math!",
    "Sure! What’s the subject?",
    "AI is evolving fast — just like my love for pizza",
    "A team of devs with too much coffee and ambition.",
    "I love digital donuts. Zero calories, 100% vibes.",
    "Did you know octopuses have 3 hearts?",
    "That’s okay. I’m here to talk if you want.",
    "How about drawing or listening to music?",
    "Only if you count beeping in binary as singing 101101",
    "Timeless. Like your best playlist."
]

def generate_conversations(num_pairs=50):
    conversations = []
    for _ in range(num_pairs):
        q = random.choice(user_questions)
        a = random.choice(bot_responses)
        conversations.append(f'"User: {q}" "Bot: {a}"')
    return "\n".join(conversations)

# Save to file or print
if __name__ == "__main__":
    num = 1000  # change this to generate more/less
    output = generate_conversations(num)
    
    with open("training_conversations.txt", "w") as f:
        f.write(output)
    
    print(f'{num} conversations written to "training_conversations.txt".')
