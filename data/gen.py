"""Generate labelled phrases: utterance -> tool call. Spans (time phrases, note text) are copied verbatim.

Train and test are split by TEMPLATE and by time phrase, so the test set never shares a sentence
pattern or a duration wording with training: passing it cannot come from memorising.
Usage: python data/gen.py   -> data/out/{train,test}.jsonl
"""
import json, random, re
from pathlib import Path

R = random.Random(7)
OUT = Path(__file__).parent / "out"

# ---- time phrases: (phrase, split). "test" phrases never appear in training. ----
def durs(split):
    d = []
    if split == "train":
        for n in (1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60, 90):
            d += [f"{n} minutes", f"{n} min", f"{n}min", f"{n} minute"]
        d += ["1 hour", "2 hours", "1.5 hours", "an hour", "half an hour", "one hour", "ten minutes", "twenty five minutes",
              "pachees minute", "das minute", "paanch minute", "bees minute", "tees minute", "chalis minute",
              "aadha ghanta", "ek ghanta", "do ghante", "dedh ghanta", "dhai ghante", "15 minute", "20 minat", "90 seconds",
              "30 seconds", "2 ghante", "1 ghanta", "pandrah minute", "paintalis minute", "sawa ghanta",
              "ek minute", "do minute", "teen minute", "char minute", "chhe minute", "saat minute", "aath minute", "nau minute", "gyarah minute", "terah minute",
              "chaudah minute", "satrah minute", "atharah minute", "unnis minute", "pachaas minute", "dedh minute", "4 hours", "6 minutes", "18 minutes", "22 min",
              "33 minutes", "55 minutes", "75 minutes", "100 minutes", "2.5 hours", "one and a half hours", "thirty minutes", "twenty minutes", "five minutes",
              "forty minutes", "sixty minutes", "ek ghante", "teen ghante", "char ghante", "paanch ghante", "10 sec", "15 seconds", "2 min", "9 minutes"]
    else:
        d = ["35 minutes", "40 min", "12 minutes", "7 min", "8 mins", "3 hours", "an hour and a half", "forty five minutes",
             "fifteen minutes", "pachas minute", "solah minute", "saath minute", "paune do ghante", "sadhe teen ghante",
             "45 sec", "1 hr", "2 hrs", "quarter of an hour", "barah minute", "ikkis minute", "50 minute", "teen ghante"]
    return d

CLOCKS = {"train": ["5 pm", "7:30 am", "shaam 5 baje", "subah 7 baje", "9 baje", "raat 11 baje", "6:15 pm", "dopahar 2 baje", "8 am"],
          "test": ["4 pm", "6:45 am", "shaam 6 baje", "subah 8 baje", "10 baje", "raat 10 baje", "saadhe paanch baje", "7 pm"]}
NOTES = {"train": ["buy milk", "call mom tomorrow", "submit the lab report", "milk aur bread lena hai", "ideas for the project",
                   "wifi password is on the router", "kal professor se milna hai", "check train tickets", "send the resume to hr",
                   "buy a notebook", "kitab library me wapas karni hai", "meeting notes from monday", "try the new recipe", "dad ka birthday gift", "fix the bug in login page",
                   "assignment 3 due friday", "gym jana hai", "recharge the phone", "return the charger"],
         "test": ["pay the electricity bill", "book a cab for friday", "dentist appointment on monday", "charger lena mat bhoolna",
                  "read chapter four", "bhai ko birthday wish karna hai"]}
LABELS = {"train": ["tea", "study", "break", "laundry", "revision", "pasta"], "test": ["maggi", "reading", "stretch", "coding"]}
REMS = {"train": ["drink water", "take medicine", "call mom", "pani peena hai", "submit assignment", "stretch", "turn off the geyser", "meeting join karna", "practice coding", "walk karna hai", "call the bank", "take a break"],
        "test": ["check the oven", "email the professor", "dawai leni hai", "go to the lab", "water the plants"]}

# ---- templates: {d}=duration {c}=clock {n}=note {l}=label {r}=reminder text. Each list is (train, test). ----
def T(train, test): return {"train": train, "test": test}
SPEC = {
 "start_timer": T(
   ["set a timer for {d}", "start a timer for {d}", "timer {d}", "{d} ka timer lagao", "{d} timer set karo", "start timer {d} please",
    "countdown {d}", "can you set timer {d}", "{d} ke liye timer chalu karo", "timer start {d}"],
   ["give me a {d} timer", "{d} ka timer laga do", "i need a timer {d}", "timer lagana {d}"]),
 "schedule_power:sleep": T(
   ["put my pc to sleep after {d}", "sleep the laptop in {d}", "suspend in {d}", "{d} baad laptop sula do", "{d} me pc sleep kar dena",
    "turn on sleep mode after {d}", "make the computer sleep after {d}", "go to sleep in {d}", "sleep after {d}", "{d} baad sleep",
    "pc ko {d} baad suspend karo", "laptop sleep mode {d} me"],
   ["after {d} put the pc to sleep", "i want sleep mode in {d}", "{d} ke baad laptop so jaye", "enable sleep after {d}"]),
 "schedule_power:poweroff": T(
   ["shut down the pc in {d}", "turn off the laptop after {d}", "{d} baad pc band kar do", "poweroff in {d}", "shutdown after {d}",
    "{d} me laptop shutdown karo", "switch off computer after {d}"],
   ["power off after {d}", "{d} baad computer band", "please shut the laptop down in {d}"]),
 "schedule_power:lock": T(
   ["lock the screen in {d}", "lock my pc after {d}", "{d} baad screen lock karo", "lock in {d}", "{d} me laptop lock kar do"],
   ["lock the computer after {d}", "{d} ke baad lock"]),
 "set_reminder": T(
   ["remind me to {r} in {d}", "remind me to {r} at {c}", "{d} baad yaad dilana {r}", "{c} pe yaad dilana {r}", "reminder {r} in {d}",
    "set a reminder to {r} at {c}", "{r} ke liye {d} baad remind karna", "remind me in {d} to {r}"],
   ["can you remind me to {r} in {d}", "{r} yaad dilana {c} pe", "remind me at {c} to {r}", "{d} me remind karo {r}"]),
 "add_note": T(
   ["note this down: {n}", "add a note {n}", "save a note: {n}", "note kar lo {n}", "yaad rakhna {n}", "make a note that {n}",
    "write down {n}", "note: {n}", "ek note likho {n}"],
   ["take a note {n}", "jot down {n}", "note likh do {n}", "remember this: {n}"]),
 "read_notes": T(
   ["read my notes", "what are my last notes", "show my recent notes", "mere notes sunao", "last notes kya hain", "read back my notes"],
   ["what did i note down", "recent notes batao", "tell me my latest notes"]),
 "focus_start": T(
   ["start a focus session", "start focus for {d}", "focus mode on", "{d} ka focus shuru karo", "pomodoro start", "begin a pomodoro",
    "focus karna hai {d}", "start pomodoro for {d}", "focus session chalu karo"],
   ["i want to focus for {d}", "start focusing", "{d} focus lagao", "pomodoro chalu karo"]),
 "focus_stop": T(["stop focus", "end the focus session", "focus band karo", "stop the pomodoro", "focus khatam"],
                 ["cancel focus session", "pomodoro band kar do"]),
 "focus_pause": T(["pause focus", "pause the pomodoro", "focus pause karo", "resume focus", "focus resume karo"],
                  ["pause my focus session", "focus ko roko"]),
 "focus_status": T(["how is my focus going", "focus status", "how many pomodoros today", "focus kaisa chal raha hai", "kitna focus hua aaj",
                    "how much time left in focus"],
                   ["what is my focus streak", "focus me kitna time bacha"]),
 "stopwatch:start": T(["start the stopwatch", "stopwatch chalu karo", "start stopwatch", "run the stopwatch", "stopwatch shuru"],
                      ["begin the stopwatch", "stopwatch on karo"]),
 "stopwatch:pause": T(["pause the stopwatch", "stop the stopwatch", "stopwatch roko", "stopwatch pause karo"],
                      ["stopwatch band karo", "halt the stopwatch"]),
 "stopwatch:reset": T(["reset the stopwatch", "stopwatch reset karo", "clear the stopwatch", "stopwatch zero karo"],
                      ["reset stopwatch please", "stopwatch ko reset kar do"]),
 "stopwatch:status": T(["what is the stopwatch at", "stopwatch kitna hua", "how long on the stopwatch", "check stopwatch"],
                       ["stopwatch status batao", "show stopwatch time"]),
 "do_not_disturb:on": T(["turn on do not disturb", "dnd on", "do not disturb for {d}", "mute notifications for {d}", "disturb mat karo {d}",
                         "notifications band karo", "dnd chalu karo {d}", "silence notifications"],
                        ["enable dnd for {d}", "{d} tak dnd on rakho", "stop notifications"]),
 "do_not_disturb:off": T(["turn off do not disturb", "dnd off", "notifications on karo", "enable notifications again", "dnd band karo"],
                         ["disable dnd", "notifications wapas chalu karo"]),
 "hotspot:on": T(["turn on the hotspot", "start wifi hotspot", "hotspot on karo", "enable hotspot", "hotspot chalu kar do"],
                 ["switch on hotspot", "wifi hotspot start karo"]),
 "hotspot:off": T(["turn off the hotspot", "stop wifi hotspot", "hotspot band karo", "disable hotspot"],
                  ["switch off hotspot", "hotspot off kar do"]),
 "network_status": T(["is my internet working", "network status", "check the network", "internet chal raha hai kya", "wifi theek hai",
                      "why is the internet slow", "net ka status batao"],
                     ["is the wifi ok", "internet kaam kar raha hai?", "how is the network"]),
 "list_scheduled": T(["what is scheduled", "list my timers", "what timers are running", "kya kya scheduled hai", "show pending reminders",
                      "mere timers dikhao"],
                     ["any timers running", "kaun se timers chal rahe hain"]),
 "cancel_scheduled:all": T(["cancel everything", "cancel all timers", "sab timers cancel karo", "clear all reminders", "cancel all scheduled"],
                           ["cancel all of them", "sab cancel kar do"]),
 "cancel_scheduled:sleep": T(["cancel the sleep timer", "don't sleep the pc anymore", "sleep cancel karo", "cancel sleep"],
                             ["stop the scheduled sleep", "sleep wala timer hata do"]),
 "cancel_scheduled:timer": T(["cancel the timer", "timer cancel karo", "stop my timer", "remove the timer"],
                             ["delete the timer", "timer hata do"]),
 "cancel_scheduled:reminder": T(["cancel the reminder", "reminder cancel karo", "remove my reminder"],
                                ["delete the reminder", "reminder hata do"]),
}
# clarify: unclear or out of scope; the model must ask instead of guessing.
CLARIFY = {"train": ["what is the weather", "send a mail to my friend", "tell me a joke", "volume up",
                     "sleep", "hello", "kya haal hai", "kuch karo", "do it", "later", "how old are you",
                     "turn it off", "call someone", "brightness kam karo", "kal", "pc"],
           "test": ["what time is it in tokyo", "make it louder", "do something", "mausam kaisa hai",
                    "shutdown", "thanks", "what can you do"]}
ASK = {"sleep": "Sleep the PC after how long?", "timer": "For how long?"}


# ---- round 2: many more sentence patterns, so the model learns the tool, not the template ----
MORE = {
 "start_timer": T(["alarm for {d}", "{d} baad batana", "wake me in {d}", "timer on {d}", "start a {d} countdown", "{d} ka countdown", "ek timer lagao {d}",
                   "set timer to {d}", "{d} ke liye timer", "need a timer, {d}", "run a timer for {d}", "{d} tak timer chalao", "timer lagao {d} ke liye",
                   "start timer for {d} for {l}", "{l} ke liye {d} ka timer"], ["can u start a timer {d}", "{d} ka timer set kar dena", "countdown timer {d} start"]),
 "schedule_power:sleep": T(["suspend the laptop after {d}", "laptop ko {d} baad sula do", "after {d} sleep karna", "sleep mode on in {d}", "{d} ke baad pc sleep",
                            "i want the pc to sleep in {d}", "set sleep timer {d}", "sleep timer {d}", "{d} me suspend", "schedule sleep for {d}",
                            "pc ko sula dena {d} baad", "turn the laptop to sleep after {d}", "put laptop in sleep mode {d} later", "{d} baad so jana laptop"],
                           ["sleep the pc {d} from now", "{d} baad pc ko sleep pe daal do", "laptop sleep after {d} please"]),
 "schedule_power:poweroff": T(["power off the pc in {d}", "{d} baad shutdown", "shut it down after {d}", "{d} me computer off kar do", "turn the laptop off in {d}",
                               "schedule shutdown {d}", "shutdown timer {d}", "laptop band karna {d} baad", "pc off after {d}"],
                              ["switch the pc off in {d}", "{d} baad laptop off"]),
 "schedule_power:lock": T(["lock screen after {d}", "{d} baad lock", "lock the laptop in {d}", "screen lock kar do {d} me", "lock it in {d}", "{d} ke baad screen lock"],
                          ["lock pc in {d}", "{d} me lock laga do"]),
 "set_reminder": T(["remind me about {r} in {d}", "yaad dilana {r} {d} baad", "{r} {c} par yaad dilana", "reminder for {r} at {c}", "ping me to {r} in {d}",
                    "don't let me forget to {r} at {c}", "{d} baad batana {r}", "set reminder {r} {d}", "remind {r} at {c}", "mujhe {c} pe {r} yaad dilana",
                    "{r} karna hai {c} pe remind karna", "remind me {r} after {d}"],
                   ["alert me to {r} in {d}", "{c} baje {r} yaad dila dena", "remind me to {r}, {d} from now"]),
 "add_note": T(["note it: {n}", "save this: {n}", "mera note: {n}", "add to notes {n}", "write this in my notes: {n}", "isse note kar lo: {n}",
                "note down {n}", "jot this down {n}", "remember: {n}", "ek note add karo: {n}", "quick note {n}", "notes me likho {n}", "yeh note kar do {n}"],
               ["put this in my notes: {n}", "note kar lena {n}", "save note {n}"]),
 "read_notes": T(["show notes", "notes dikhao", "read me my notes", "meri last notes padho", "what did i write down", "latest notes please", "recent notes", "notes padh ke sunao"],
                 ["read my last note", "pichle notes batao"]),
 "focus_start": T(["focus time", "start focus mode", "begin focus for {d}", "let me focus for {d}", "focus shuru karo", "pomodoro shuru {d}", "start a {d} focus",
                   "focus chalu", "focus session start karo {d}", "start a work session", "padhai ka focus lagao", "i need to concentrate for {d}"],
                  ["turn on focus mode", "focus mode lagao {d}", "start my pomodoro"]),
 "focus_stop": T(["stop the focus", "end focus now", "focus stop karo", "i am done focusing", "quit focus mode", "focus mode off", "focus rok do", "pomodoro stop"],
                 ["finish the focus session", "focus mode band karo"]),
 "focus_pause": T(["hold the focus session", "take a break from focus", "focus ko pause karo", "pomodoro pause", "unpause focus", "continue focus", "focus phir se chalu karo"],
                  ["pause pomodoro for now", "focus thoda rok do"]),
 "focus_status": T(["focus time left", "how long left in focus", "focus kitna bacha hai", "pomodoro status", "how many sessions today", "focus progress", "aaj kitne pomodoro hue"],
                   ["how is the focus session", "focus ka status"]),
 "stopwatch:start": T(["begin stopwatch", "stopwatch lagao", "start timing", "stopwatch start karo", "run stopwatch", "time shuru karo"], ["turn on the stopwatch", "stopwatch chalao"]),
 "stopwatch:pause": T(["freeze the stopwatch", "stopwatch ruko", "pause stopwatch", "stop timing", "stopwatch rok do"], ["stopwatch ko pause kar do", "stop the watch"]),
 "stopwatch:reset": T(["stopwatch ko reset karo", "zero the stopwatch", "restart stopwatch from zero", "reset timing", "stopwatch clear karo"], ["wipe the stopwatch", "stopwatch reset kar dena"]),
 "stopwatch:status": T(["stopwatch time batao", "what does the stopwatch say", "how much time on stopwatch", "stopwatch dikhao", "elapsed time"], ["tell me the stopwatch time", "stopwatch me kitna hai"]),
 "do_not_disturb:on": T(["dnd lagao", "do not disturb mode on", "mute all notifications {d}", "no notifications for {d}", "silent mode on karo", "pause notifications for {d}",
                         "notifications mute karo {d} ke liye", "dnd on for {d}", "quiet mode {d}", "disturb nahi karna {d}", "turn on dnd"],
                        ["don't disturb me for {d}", "notifications off kar do {d}", "dnd mode chalu"]),
 "do_not_disturb:off": T(["turn dnd off", "do not disturb band karo", "unmute notifications", "notifications chalu karo", "disable do not disturb", "silent mode off", "dnd hata do"],
                         ["switch off dnd", "notifications wapas on"]),
 "hotspot:on": T(["hotspot chalu karo", "turn the hotspot on", "share my wifi", "start the hotspot", "hotspot on", "mobile hotspot start karo", "enable wifi sharing"], ["activate hotspot", "hotspot lagao"]),
 "hotspot:off": T(["hotspot off", "hotspot band", "stop sharing wifi", "turn the hotspot off", "hotspot hata do", "wifi sharing band karo"], ["kill the hotspot", "hotspot bandh karo"]),
 "network_status": T(["internet check karo", "is the wifi working", "net chal raha hai?", "check my connection", "network theek hai kya", "am i online", "internet slow hai kya", "connection status", "wifi ka haal batao"],
                     ["is the network fine", "net theek hai"]),
 "list_scheduled": T(["show my timers", "kitne timers chal rahe hain", "what reminders do i have", "list scheduled things", "pending timers dikhao", "kya kya set hai", "any reminders pending"],
                     ["what is pending", "timers list karo"]),
 "cancel_scheduled:all": T(["cancel all of it", "clear everything scheduled", "sab hata do", "remove all timers", "sab kuch cancel", "delete all reminders and timers"], ["cancel it all", "sab reminders hata do"]),
 "cancel_scheduled:sleep": T(["don't sleep", "sleep timer hata do", "remove the sleep timer", "cancel the scheduled sleep", "pc ko sulana mat", "sleep wala cancel karo"], ["stop the sleep timer", "sleep cancel kar do"]),
 "cancel_scheduled:timer": T(["stop the countdown", "timer rok do", "cancel my timer", "timer band karo", "kill the timer", "timer wala cancel karo"], ["clear the timer", "timer cancel kar dena"]),
 "cancel_scheduled:reminder": T(["no more reminder", "reminder hata do", "cancel that reminder", "reminder rok do", "forget the reminder"], ["drop the reminder", "reminder cancel kar do"]),
}
for k, v in MORE.items():
    SPEC[k]["train"] += v["train"]; SPEC[k]["test"] += v["test"]
CLARIFY["train"] += ["i am just checking", "are you listening to me", "can you hear me", "testing testing", "hello can you hear me", "what are you doing",
                     "ok so", "thank you", "good job", "that was great", "wait a second", "let me think", "i am testing this", "is it working", "say something",
                     "hello jarvis", "how are you doing", "what is going on", "tell me something", "are you there", "are you listening", "i am just checking if you work",
                     "okay now listen to me", "hmm let me see", "yaar sun", "kya kar rahe ho", "sun raha hai kya", "theek hai", "acha", "ek minute ruko"]
CLARIFY["train"] += ["what's the temperature", "who are you", "sing a song", "what day is it", "search google", "turn up the volume", "make it brighter",
                     "send a message", "take a screenshot", "kitne baje hain", "mausam batao", "kya chal raha hai", "good morning",
                     "note", "focus", "stopwatch kya hai", "delete my files", "restart", "log out", "what is two plus two", "translate this", "play a video", "close the window"]
CLARIFY["test"] += ["just testing if you work", "do you hear me", "okay then", "that's cool", "why is it not working", "hey are you awake", "can you hear something",
                    "sun rahe ho", "kuch bolo", "dekhte hain"]
CLARIFY["test"] += ["what's the weather today", "tell me about yourself", "kuch bhi", "play something", "sab theek hai?", "how are you"]

# ---- round 3: media and sites ----
MEDIA = {
 "open_app": T(["open {a}", "{a} kholo", "launch {a}", "start {a}", "{a} open karo", "{a} chalu karo", "open the {a} app", "can you open {a}", "open up {a} please",
                "go to {a}", "take me to {a}", "{a} khol do", "sir open {a}", "{a} app kholo", "i want to use {a}", "run {a}", "{a} start karo", "please open {a}"],
               ["could you launch {a}", "{a} chala do", "open {a} for me", "mujhe {a} kholna hai", "bring up {a}"]),
 "play_music": T(["play {q} on youtube", "{q} bajao", "play {q}", "search {q} on youtube", "youtube pe {q} chalao", "play some {q}", "{q} sunao", "i want to listen to {q}",
                  "{q} gaana chalao", "put on {q}", "play {q} music", "{q} laga do"],
                 ["can you play {q}", "{q} play karo", "play me some {q}", "{q} suna do", "youtube par {q} lagao"]),
 "media_control:play": T(["play the music", "resume music", "music play karo", "gaana chalao", "play some music", "resume", "continue the song", "gaana chalu karo"],
                         ["resume the music", "gaana phir se chalao", "start the music"]),
 "media_control:pause": T(["pause the music", "pause", "gaana roko", "music pause karo", "stop the music", "pause the song", "gaana band karo", "pause the video"],
                          ["pause playback", "music roko", "stop the song"]),
 "media_control:next": T(["next song", "skip this song", "agla gaana", "next track", "skip", "next gaana lagao", "play the next one"], ["skip the track", "agla gaana lagao", "go to the next song"]),
 "media_control:previous": T(["previous song", "go back a song", "pichla gaana", "previous track", "play the last song again", "pehla wala gaana lagao"], ["go to the previous track", "pichla gaana lagao"]),
}
# a timer asked for as a "stopwatch" or "alarm", and timers with no time yet (Tommy then asks "What time, sir?")
SPEC["start_timer"]["train"] += ["start a {d} stopwatch", "{d} ka stopwatch lagao", "set an alarm for {d}", "alarm {d} baad", "{d} ka alarm lagao",
                                 "set a {d} alarm", "can you start a {d} stopwatch timer"]
SPEC["start_timer"]["test"] += ["start a {d} stopwatch please", "alarm laga do {d} ka"]
SPEC["set_reminder:asktext"] = T(["set a reminder", "reminder lagao", "remind me", "ek reminder set karo", "can you set a reminder", "reminder set kar do",
                                  "remind me something", "add a reminder please"], ["add a reminder", "reminder laga do", "yaad dilao"])
SPEC["start_timer:asktime"] = T(["set a timer", "set an alarm", "start a timer", "timer lagao", "alarm lagao", "ek timer lagao", "can you set a timer"],
                                ["timer chalu karo", "start an alarm", "put a timer"])
for k, v in MEDIA.items():
    SPEC[k] = v
def installed_names():
    """This PC's own app names, read at generation time (never stored in the repository)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from tommy.apps import apps
        return [x.name.lower() for x in apps() if 3 <= len(x.name) <= 24 and x.name.isascii()]
    except Exception:
        return []


APP_TEST = ["notion", "blender", "zoom", "reddit", "audacity", "postman", "figma", "signal", "krita", "google docs"]
APPS = {"train": [n for n in dict.fromkeys(["youtube", "whatsapp", "gmail", "github", "chatgpt", "maps", "spotify", "vs code", "terminal", "files", "discord", "telegram",
                  "firefox", "chrome", "calculator", "settings", "instagram", "classroom", "netflix", "linkedin", "steam", "obs", "gimp", "word", "excel", "powerpoint",
                  "camera", "gallery", "calendar", "my browser", "amazon", "wikipedia", "google drive", "you 2", "your tube", "whats app", "insta", "class room",
                  "leet code", "twitter", "photos", "music player", "text editor", "task manager"] + installed_names()) if n not in APP_TEST],
        "test": APP_TEST}
QUERIES = {"train": ["lofi", "arijit singh songs", "kishore kumar", "relaxing music", "workout music", "imagine dragons", "punjabi songs", "study music", "old hindi songs", "coldplay", "bollywood hits", "arctic monkeys"],
           "test": ["jazz", "ar rahman", "piano music", "sidhu moose wala", "shape of you", "chill beats"]}

def noise(s):
    """Typos, dropped punctuation and Hinglish respellings, applied to everything but the copied span."""
    if R.random() < 0.15: s = s.replace(" ", "  ", 1)
    if R.random() < 0.12: s = s.upper() if R.random() < 0.2 else s.capitalize()
    swaps = [("karo", "kro"), ("kar do", "krdo"), ("please", "plz"), ("laptop", "lappy"), ("minutes", "mins"), ("baad", "bad"),
             ("shuru", "shuroo"), ("the ", "")]
    if R.random() < 0.25:
        a, b = R.choice(swaps)
        if a in s and "{" not in s: s = s.replace(a, b, 1)
    return s.strip()

def make(split, per_template):
    rows = []
    D, C, N, L, RM = durs(split), CLOCKS[split], NOTES[split], LABELS[split], REMS[split]
    for key, spec in SPEC.items():
        tool, _, variant = key.partition(":")
        for tpl in spec[split]:
            for _ in range(per_template * (2 if tool == "open_app" else 1)):
                slots = {"a": R.choice(APPS[split]), "q": R.choice(QUERIES[split]), "d": R.choice(D), "c": R.choice(C), "n": R.choice(N), "l": R.choice(L), "r": R.choice(RM)}
                text = tpl.format(**slots)
                args = {}
                if tool == "start_timer":
                    args = {} if variant == "asktime" else {"duration": slots["d"]}
                elif tool == "schedule_power":
                    args = {"action": variant, "after": slots["d"]}
                elif tool == "set_reminder":
                    args = {} if variant == "asktext" else {"text": slots["r"], "when": slots["c"] if "{c}" in tpl else slots["d"]}
                elif tool == "add_note":
                    args = {"text": slots["n"]}
                elif tool == "focus_start" and "{d}" in tpl:
                    args = {"duration": slots["d"]}
                elif tool == "stopwatch":
                    args = {"action": variant}
                elif tool == "do_not_disturb":
                    args = {"state": variant, **({"duration": slots["d"]} if "{d}" in tpl else {})}
                elif tool == "open_app":
                    args = {"name": slots["a"]}
                elif tool == "play_music":
                    args = {"query": slots["q"]}
                elif tool == "media_control":
                    args = {"action": variant}
                elif tool == "hotspot":
                    args = {"state": variant}
                elif tool == "cancel_scheduled":
                    args = {"what": variant}
                rows.append({"text": noise(text) if split == "train" else text, "tool": tool, "args": args})
    for phrase in CLARIFY[split]:
        for _ in range(per_template):
            rows.append({"text": phrase, "tool": "clarify", "args": {"question": ASK.get(phrase.split()[0] if phrase.split()[0] in ASK else "", "Sorry, can you say that differently?")}})
    # dedupe
    seen, out = set(), []
    for r in rows:
        k = (r["text"], r["tool"], json.dumps(r["args"], sort_keys=True))
        if k not in seen: seen.add(k); out.append(r)
    R.shuffle(out)
    return out

if __name__ == "__main__":
    tr, te = make("train", 6), make("test", 5)
    seen_train = {r["text"].lower() for r in tr}
    dropped = [r["text"] for r in te if r["text"].lower() in seen_train]
    te = [r for r in te if r["text"].lower() not in seen_train]      # a test phrase that also appears in training is not held out
    if dropped: print("dropped from test (also in train):", len(dropped), sorted(set(dropped))[:6])
    assert not {r["text"].lower() for r in tr} & {r["text"].lower() for r in te}
    for name, rows in (("train", tr), ("test", te)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(name, len(rows), "rows;", len({r['tool'] for r in rows}), "tools")
