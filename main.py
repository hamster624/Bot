import discord
from discord.ext import commands
from discord import app_commands
import logging
from dotenv import load_dotenv
import os
import time
import json
from flask import Flask
from threading import Thread
import math
import re

# ---------------------
# Flask Keep-Alive
# ---------------------
app = Flask('')
@app.route('/')
def home():
    return "I'm alive"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

keep_alive()

# ---------------------
# Logging Setup
# ---------------------
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ],
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s'
)

# ---------------------
# Bot Setup
# ---------------------
load_dotenv()
token = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
STATE_FILE = "counting_state.json"
state = {}

# ---------------------
# State Management
# ---------------------
def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_state():
    global state
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {}

def get_guild_state(guild_id):
    return state.get(str(guild_id), {
        "counting_channel_id": None,
        "current_count": 0,
        "last_counter": None
    })

def set_guild_state(guild_id, counting_channel_id, current_count, last_counter):
    state[str(guild_id)] = {
        "counting_channel_id": counting_channel_id,
        "current_count": current_count,
        "last_counter": last_counter
    }
    save_state()

# ---------------------
# Bot Events
# ---------------------
@bot.event
async def on_ready():
    load_state()
    await bot.tree.sync()
    print(f"✅ Logged in as {bot.user}")

# ---------------------
# Counting Commands
# ---------------------
@bot.command()
@commands.has_permissions(administrator=True)
async def setcount(ctx, number: int):
    guild_id = ctx.guild.id
    guild_state = get_guild_state(guild_id)
    if guild_state["counting_channel_id"] is None:
        await ctx.send("⚠ No counting channel set! Use `!setcounting #channel` first.")
        return
    set_guild_state(guild_id, guild_state["counting_channel_id"], number, guild_state["last_counter"])
    await ctx.send(f"✅ Current count for {ctx.guild.name} has been set to **{number}**.")

@bot.command()
@commands.has_permissions(administrator=True)
async def setcounting(ctx, channel: discord.TextChannel):
    guild_id = ctx.guild.id
    set_guild_state(guild_id, channel.id, 0, None)
    await ctx.send(f"✅ Counting channel set to {channel.mention}. Counter reset to 0.")

@bot.command()
async def current(ctx):
    guild_state = get_guild_state(ctx.guild.id)
    await ctx.send(
        f"🔹 Current count: **{guild_state['current_count']}**\n"
        f"📢 Channel: <#{guild_state['counting_channel_id']}>"
        if guild_state['counting_channel_id'] else "⚠ No counting channel set!"
    )

def rewrite_exponent(expr: str) -> str:
    return re.sub(r'(\d+|\w+|\([^()]*\))\s*\*\*\s*(\d+|\w+|\([^()]*\))',r'pow(\1, \2)', expr)
# ---------------------
# Counting Logic
# ---------------------
@bot.event
async def on_message(message):
    if message.author.bot or not message.guild:
        return

    guild_id = message.guild.id
    guild_state = get_guild_state(guild_id)
    counting_channel_id = guild_state['counting_channel_id']
    current_count = guild_state['current_count']
    last_counter = guild_state['last_counter']

    if counting_channel_id and message.channel.id == counting_channel_id:
        content = message.content.strip()
        if not content: return
        content = content.replace("^", "**")
        content = rewrite_exponent(content)
        first_token = content.split()[0]

        safe_globals = {
            "__builtins__": {},
            "math": math,
            "tetration": tetration,
            "tetr": tetration,
            "pent": pent,
            "hex": hex,
            "hept": hept,
            "arrow": arrow,
            "fact": fact,
            "factorial": factorial,
            "gamma": gamma,
            "slog": slog,
            "plog": plog,
            "hlog": hlog,
            "hyper_log": hyper_log,
            "addlayer": addlayer,
            "add": add,
            "sub": sub,
            "mul": mul,
            "div": div,
            "pow": pow,
            "power": power,
            "exp": exp,
            "lambertw": lambertw,
            "root": root,
            "sqrt": sqrt,
            "eq": eq,
            "lt": lt,
            "gte": gte,
            "gt": gt,
            "lte": lte,
            "min": min,
            "max": max,
            "floor": floor,
            "ceil": ceil,
            "log": log,
            "ln": ln,
            "logbase": logbase
        }

        try:
            value = eval(first_token, safe_globals, {})
            value = math.floor(float(value) + 0.5) if value >= 0 else math.ceil(float(value) - 0.5)
        except Exception:
            return

        if message.author.id == last_counter:
            await message.channel.send(
                f"❌ {message.author.mention}, you counted twice in a row! Count resets to 1."
            )
            current_count = 0
            last_counter = None
            set_guild_state(guild_id, counting_channel_id, current_count, last_counter)
            return

        if value == current_count + 1:
            current_count += 1
            last_counter = message.author.id
            await message.add_reaction("✅")
        else:
            await message.channel.send(
                f"❌ {message.author.mention} failed!\n"
                f"➡ The next number is now **1**.\n"
                f"🔹 Last successful number was **{current_count}**."
            )
            current_count = 0
            last_counter = None

        set_guild_state(guild_id, counting_channel_id, current_count, last_counter)

    await bot.process_commands(message)

# ---------------------
# Calculator Commands (!calc + /calc)
# ---------------------
@bot.command()
async def guide(ctx):
    formats = ["format","array","hyper_e","suffix", "string"]
    operations = ["arrow (arrow)","hept (heptation)","hex (hexation)","pent (pentation)", "tetr (tetration)", "pow (power)", "exp", "root", "sqrt", "addlayer",
                  "log", "ln", "logbase", "slog", "plog", "hlog", "hyper_log", "lambertw",
                  "fact (factorial)", "gamma",
                  "add (addition)", "sub (subtract)", "mul (multiply)", "div (division)",
                  "eq", "lt", "gt", "gte", "lte", "min", "max",
                  "floor", "ceil"]
    help_message = "**📘 !calc Help**\n\n"
    help_message += "**Available Formats:**\n" + ", ".join(formats) + "\n\n"
    help_message += "**Supported Operations:**\n" + ", ".join(operations) + "\n\n"
    help_message += "To use arrow the 1st number is base 2nd is arrows 3rd is height so arrow(10,4,10)=10^^^^10. Max 40 arrows.\n"
    help_message += "To use hyperlog the 1st number is the number you want to log and 2nd is arrows for example if log10(x) = 10^x and we solve for x then hyper_log(x, 5) = 10^^^^^x where we solve for x. Make sure x is either in hyper_e or as an array. \n"
    help_message += "Usage: `/calc <expression> [format]`\n"
    help_message += "Number Usage: for numbers below 2^1024 you can use float, but after they go higher use the 'correct' formats output so 1F10=(10^)^9 10 and make sure to put them as strings."
    await ctx.send(help_message)

@bot.command()
async def calc(ctx, *, expression: str):
    formats = {
        "format": format,
        "string": string,
        "hyper_e": hyper_e,
        "suffix": suffix,
        "array": correct,
    }
    try:
        tokens = expression.split()
        fmt_name = "format"
        if tokens and tokens[-1].lower() in formats:
            fmt_name = tokens[-1].lower()
            tokens = tokens[:-1]

        expr = " ".join(tokens).replace("^", "**")
        expr = rewrite_exponent(expr)

        safe_globals = {
            "__builtins__": {},
            "math": math,
            "tetration": tetration,
            "tetr": tetration,
            "pent": pent,
            "hex": hex,
            "hept": hept,
            "arrow": arrow,
            "fact": fact,
            "factorial": factorial,
            "gamma": gamma,
            "slog": slog,
            "plog": plog,
            "hlog": hlog,
            "hyper_log": hyper_log,
            "addlayer": addlayer,
            "add": add,
            "sub": sub,
            "mul": mul,
            "div": div,
            "pow": pow,
            "power": power,
            "exp": exp,
            "lambertw": lambertw,
            "root": root,
            "sqrt": sqrt,
            "eq": eq,
            "lt": lt,
            "gte": gte,
            "gt": gt,
            "lte": lte,
            "min": min,
            "max": max,
            "floor": floor,
            "ceil": ceil,
            "log": log,
            "ln": ln,
            "logbase": logbase
        }
        start_time = time.time()
        value = eval(expr, safe_globals, {})
        result = formats[fmt_name](value)
        elapsed = time.time() - start_time

        await ctx.reply(
            f"**Result:** ```{result}```\n⏱ Evaluated in {elapsed:.6f} seconds",
            mention_author=False
        )
    except Exception as e:
        await ctx.reply(f"❌ Error: `{e}`", mention_author=False)

# ---------------------
# Guide Command (slash)
# ---------------------
@bot.tree.command(name="guide", description="Shows help for the calculator")
async def guide_slash(interaction: discord.Interaction):
    formats = ["format","array","hyper_e","suffix", "string"]
    operations = ["arrow (arrow)","hept (heptation)","hex (hexation)","pent (pentation)", "tetr (tetration)", "pow (power)", "exp", "root", "sqrt", "addlayer",
                  "log", "ln", "logbase", "slog", "plog", "hlog", "hyper_log", "lambertw",
                  "fact (factorial)", "gamma",
                  "add (addition)", "sub (subtract)", "mul (multiply)", "div (division)",
                  "eq", "lt", "gt", "gte", "lte", "min", "max",
                  "floor", "ceil"]
    help_message = "**📘 !calc Help**\n\n"
    help_message += "**Available Formats:**\n" + ", ".join(formats) + "\n\n"
    help_message += "**Supported Operations:**\n" + ", ".join(operations) + "\n\n"
    help_message += "To use arrow the 1st number is base 2nd is arrows 3rd is height so arrow(10,4,10)=10^^^^10. Max 40 arrows.\n"
    help_message += "To use hyperlog the 1st number is the number you want to log and 2nd is arrows for example if log10(x) = 10^x and we solve for x then hyper_log(x, 5) = 10^^^^^x where we solve for x. Make sure x is either in hyper_e or as an array. \n"
    help_message += "Usage: `/calc <expression> [format]`\n"
    help_message += "Number Usage: for numbers below 2^1024 you can use float, but after they go higher use the 'correct' formats output so 1F10=(10^)^9 10 and make sure to put them as strings."
    await interaction.response.send_message(help_message)

# ---------------------
# Calculator Command (/calc)
# ---------------------
@bot.tree.command(name="calc", description="Evaluate an expression with format support")
@app_commands.describe(
    expression="Expression to evaluate",
    fmt="Optional output format"
)
async def calc_slash(
    interaction: discord.Interaction,
    expression: str,
    fmt: str = "format"
):
    formats = {
        "format": format,
        "string": string,
        "hyper_e": hyper_e,
        "suffix": suffix,
        "array": correct,
    }

    await interaction.response.defer(thinking=True)

    try:
        expr = expression.replace("^", "**")
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "tetration": tetration,
            "tetr": tetration,
            "pent": pent,
            "hex": hex,
            "hept": hept,
            "arrow": arrow,
            "fact": fact,
            "factorial": factorial,
            "gamma": gamma,
            "slog": slog,
            "plog": plog,
            "hlog": hlog,
            "hyper_log": hyper_log,
            "addlayer": addlayer,
            "add": add,
            "sub": sub,
            "mul": mul,
            "div": div,
            "pow": pow,
            "power": power,
            "exp": exp,
            "lambertw": lambertw,
            "root": root,
            "sqrt": sqrt,
            "eq": eq,
            "lt": lt,
            "gte": gte,
            "gt": gt,
            "lte": lte,
            "min": min,
            "max": max,
            "floor": floor,
            "ceil": ceil,
            "log": log,
            "ln": ln,
            "logbase": logbase
        }
        start_time = time.time()
        expr = expression.replace("^", "**")
        expr = rewrite_exponent(expr)
        value = eval(expr, safe_globals, {})
        fmt_name = fmt.lower()
        if fmt_name not in formats:
            fmt_name = "format"

        result = formats[fmt_name](value)
        elapsed = time.time() - start_time

        await interaction.followup.send(
            f"**Result:** ```{result}```\n⏱ Evaluated in {elapsed:.6f} seconds"
        )
    except:
        await interaction.followup.send(f"❌ Error: `{e}`")
import math
#--Edtiable things--
decimals = 6 # How many decimals (duh). Max 16
max_suffix = 3_000_003 # At how much 10^x it goes from being suffix to scientific Example: e1000 -> e1K
FirstOnes = ["", "U", "D", "T", "Qd", "Qn", "Sx", "Sp", "Oc", "No"]
SecondOnes = ["", "De", "Vt", "Tg", "qg", "Qg", "sg", "Sg", "Og", "Ng"]
ThirdOnes = ["", "Ce", "Du", "Tr", "Qa", "Qi", "Se", "Si", "Ot", "Ni"]
MultOnes = [
    "", "Mi", "Mc", "Na", "Pi", "Fm", "At", "Zp", "Yc", "Xo", "Ve", "Me", "Due", 
    "Tre", "Te", "Pt", "He", "Hp", "Oct", "En", "Ic", "Mei", "Dui", "Tri", "Teti", 
    "Pti", "Hei", "Hp", "Oci", "Eni", "Tra", "TeC", "MTc", "DTc", "TrTc", "TeTc", 
    "PeTc", "HTc", "HpT", "OcT", "EnT", "TetC", "MTetc", "DTetc", "TrTetc", "TeTetc", 
    "PeTetc", "HTetc", "HpTetc", "OcTetc", "EnTetc", "PcT", "MPcT", "DPcT", "TPCt", 
    "TePCt", "PePCt", "HePCt", "HpPct", "OcPct", "EnPct", "HCt", "MHcT", "DHcT", 
    "THCt", "TeHCt", "PeHCt", "HeHCt", "HpHct", "OcHct", "EnHct", "HpCt", "MHpcT", 
    "DHpcT", "THpCt", "TeHpCt", "PeHpCt", "HeHpCt", "HpHpct", "OcHpct", "EnHpct", 
    "OCt", "MOcT", "DOcT", "TOCt", "TeOCt", "PeOCt", "HeOCt", "HpOct", "OcOct", 
    "EnOct", "Ent", "MEnT", "DEnT", "TEnt", "TeEnt", "PeEnt", "HeEnt", "HpEnt", 
    "OcEnt", "EnEnt", "Hect", "MeHect"
]
#--End of editable things--
MAX_SAFE_INT = 2**53 - 1
MAX_LOGP1_REPEATS = 48
_log10 = math.log10
# You can ignore these, these are only to help the code
def correct(x):
    if isinstance(x, (int, float)): 
        return correct([0 if x >= 0 else 1, abs(x)])

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("E") or s.startswith("-E"): return from_hyper_e(s)
        return fromString(s)

    if isinstance(x, list):
        arr = x[:]
        if not arr: return [0, 0]
        if len(arr) == 1: return [0 if arr[0] >= 0 else 1, abs(arr[0])]
        if arr[0] not in (0, 1): raise ValueError("First element must be 0 (positive) or 1 (negative) (array:{arr})")

        for i in range(1, len(arr)):
            if isinstance(arr[i], str):
                try: arr[i] = float(arr[i])
                except ValueError: raise ValueError(f"Element at index {i} must be a number (array:{arr})")
            elif not isinstance(arr[i], (int, float)): raise ValueError(f"Element at index {i} must be a number (array:{arr})")
            if arr[i] < 0: raise ValueError(f"Element at index {i} must be positive (array:{arr})")

        changed = True
        while changed:
            changed = False
            for i in range(len(arr)-1, 0, -1):
                if arr[i] > MAX_SAFE_INT:
                    L = _log10(arr[i])
                    if i == 1:
                        arr[1] = L
                        if len(arr) > 2: arr[2] += 1
                        else: arr.append(1)
                    else:
                        arr[1] = L
                        for j in range(2, i):
                            arr[j] = 1
                        if i == 2: arr[2] = 1
                        else: arr[i] = 0
                        if i == len(arr) - 1: arr.append(1)
                        else: arr[i+1] += 1
                    changed = True
                    break

        for i in range(1, len(arr)):
            if isinstance(arr[i], float) and arr[i] <= MAX_SAFE_INT and arr[i].is_integer():
                arr[i] = int(arr[i])

        while len(arr) >= 3 and arr[2] >= 1 and arr[1] <= _log10(MAX_SAFE_INT):
            collapsed_val = 10 ** arr[1]
            if arr[2] == 1:
                if len(arr) == 3: arr = [arr[0], collapsed_val]
                else: arr = [arr[0], collapsed_val, 0] + arr[3:]
            else: arr = [arr[0], collapsed_val, arr[2]-1] + arr[3:]

        if len(arr) > 3 and arr[2] == 0:
            z = 0
            i = 2
            while i < len(arr) and arr[i] == 0:
                z += 1
                i += 1
            if i == len(arr): arr.append(1)
            if arr[i] == 0: arr[i] = 0
            else: arr[i] -= 1
            num_eights = 1 if z == 1 or z == 2 else (z - 1)
            a1 = arr[1]
            if isinstance(a1, float) and a1.is_integer(): a1 = a1
            mid = [8] * num_eights + [a1 - 2]
            arr = arr[:2] + mid + arr[i:]

        while len(arr) > 2 and arr[-1] == 0: arr.pop(-1)

        return arr
    raise TypeError("Unsupported type for correct")
def fromString(s):
    s = s.strip()
    sign = 0
    if s.startswith("-"):
        sign = 1
        s = s[1:].lstrip()

    n = len(s)
    i = 0
    ops = []
    base = None

    def read_int(start):
        j = start
        while j < n and s[j].isdigit():
            j += 1
        if j == start:
            return None, start
        return int(s[start:j]), j

    def read_float(start):
        j = start
        dot = False
        while j < n and (s[j].isdigit() or (s[j] == '.' and not dot)):
            dot = dot or (s[j] == '.')
            j += 1
        if j == start:
            return None, start
        return float(s[start:j]), j

    while i < n:
        if s[i].isspace():
            i += 1
            continue

        if s.startswith("(10{", i):
            j = i + 4
            D, j2 = read_int(j)
            if D is not None and j2 < n and s[j2] == "}":
                j2 += 1
                if j2 < n and s[j2] == ")":
                    j2 += 1
                if j2 < n and s[j2] == "^":
                    j2 += 1
                    M, j3 = read_int(j2)
                    if M is None:
                        M, j3 = read_float(j2)
                    if M is not None:
                        ops.append(M)
                        i = j3
                        continue

        if s.startswith("(10", i):
            j = i + 3
            carets = 0
            while j < n and s[j] == "^":
                carets += 1
                j += 1
            if carets >= 1:
                if j < n and s[j] == ")":
                    j += 1
                if j < n and s[j] == "^":
                    j += 1
                    M, j2 = read_int(j)
                    if M is None:
                        M, j2 = read_float(j)
                    if M is not None:
                        ops.append(M)
                        i = j2
                        continue

        if s.startswith("10{", i):
            j = i + 3
            D, j2 = read_int(j)
            if D is not None and j2 < n and s[j2] == "}":
                ops.append(1)
                i = j2 + 1
                continue

        if s.startswith("10", i):
            j = i + 2
            carets = 0
            while j < n and s[j] == "^":
                carets += 1
                j += 1
            if carets >= 2:
                ops.append(1)
                i = j
                continue

        if s[i] == "e":
            j = i
            layer = 0
            while j < n and s[j] == "e":
                layer += 1
                j += 1
            num, j2 = read_int(j)
            if num is None: num, j2 = read_float(j)
            if num is not None:
                base = ('e', layer, num)
                i = j2
                continue
        num, j = read_int(i)
        if num is None: num, j = read_float(i)
        if num is not None:
            if base is None:
                base = ('num', num)
            i = j
            continue
        i += 1
    if base is None: raise ValueError("fromString: no base (e.. or number) found")
    if base[0] == 'e': arr = [sign, base[2], base[1]]
    else: arr = [sign, base[1]]
    for m in reversed(ops): arr.append(m)
    return correct(arr)

def from_hyper_e(s):
    if not (s.startswith("E") or s.startswith("-E")):
        raise ValueError("Not a hyper_e string")
    sign = 0
    if s.startswith("-E"):
        sign = 1
        payload = s[2:]
    else: payload = s[1:]
    if payload == "":
        raise ValueError("Invalid/empty hyper_e payload")

    parts = payload.split("#")
    nums = []
    for i, p in enumerate(parts):
        if "." in p:
            val = float(p)
            if val.is_integer(): val = int(val)
        else:
            try:
                val = int(p)
            except ValueError:
                val = float(p)
                if val.is_integer(): val = int(val)

        if i >= 2:
            if isinstance(val, int): val = val - 1
            else:
                val = val - 1
                if val.is_integer(): val = int(val) # this is to remove the .0 from floats

        nums.append(val)
    return correct([sign] + nums)

def compare(a, b):
    A = correct(a)
    B = correct(b)
    if A[0] != B[0]: return -1 if A[0] == 1 else 1
    sign = -1 if A[0] == 1 else 1
    A_layer = len(A) - 2 if len(A) > 2 else 0
    B_layer = len(B) - 2 if len(B) > 2 else 0
    if A_layer != B_layer: return sign * (1 if A_layer > B_layer else -1)
    min_len = min(len(A), len(B))
    for i in range(1, min_len + 1):
        Ai = A[-i]
        Bi = B[-i]
        if Ai != Bi:
            return sign * (1 if Ai > Bi else -1)
    if len(A) != len(B): return sign * (1 if len(A) > len(B) else -1)
    return 0

def neg(x):
    arr = correct(x)
    flipped = arr[:]
    flipped[0] = 1 - arr[0]
    return flipped
# Everything after this is from https://github.com/cloudytheconqueror/letter-notation-format
def _to_pair_array(arr):
    if not arr: return [[0, 0]]
    if isinstance(arr[0], (list, tuple)) and len(arr[0]) == 2: return [ [int(h), float(v)] for h, v in arr ]
    if not isinstance(arr, (list, tuple)): raise TypeError("polarize: unsupported array type")
    if len(arr) == 1: return [[0, float(arr[0])]]
    base_val = float(arr[1])
    pairs = [[0, base_val]]
    for i in range(2, len(arr)): pairs.append([i-1, float(arr[i])])
    return pairs

def polarize(array, smallTop=False):
    try: array = correct(array)
    except: pass
    pairs = _to_pair_array(array)
    if len(pairs) == 0:
        pairs = [[0, 0]]

    bottom = pairs[0][1] if pairs[0][0] == 0 else 10
    top = 0
    height = 0

    if len(pairs) <= 1 and pairs[0][0] == 0:
        if smallTop:
            while bottom >= 10:
                bottom = _log10(bottom)
                top += 1
                height = 1
    else:
        elem = 1 if pairs[0][0] == 0 else 0
        top = pairs[elem][1]
        height = int(pairs[elem][0])

        while (bottom >= 10) or (elem < len(pairs)) or (smallTop and top >= 10):
            if bottom >= 10:
                if height == 1:
                    bottom = _log10(bottom)
                    if bottom >= 10:
                        bottom = _log10(bottom)
                        top += 1
                elif height < MAX_LOGP1_REPEATS:
                    if bottom >= 1e10: bottom = _log10(_log10(_log10(bottom))) + 2
                    else: bottom = _log10(_log10(bottom)) + 1
                    for _i in range(2, height):
                        bottom = _log10(bottom) + 1
                else: bottom = 1
                top += 1
            else:
                if elem == len(pairs) - 1 and pairs[elem][0] == height and not (smallTop and top >= 10): break
                bottom = _log10(bottom) + top
                height += 1
                if elem < len(pairs) and height > pairs[elem][0]: elem += 1
                if elem < len(pairs):
                    if height == pairs[elem][0]:
                        top = pairs[elem][1] + 1
                    elif bottom < 10:
                        diff = pairs[elem][0] - height
                        if diff < MAX_LOGP1_REPEATS:
                            for _ in range(diff):
                                bottom = _log10(bottom) + 1
                        else: bottom = 1
                        height = pairs[elem][0]
                        top = pairs[elem][1] + 1
                    else: top = 1
                else: top = 1
    return {"bottom": bottom, "top": top, "height": int(height)}

def array_search(arr, height):
    pairs = _to_pair_array(arr)
    for h, v in pairs:
        if h == height: return v
        if h > height: break
    return 0 if height > 0 else 10

def set_to_zero(arr, height):
    if isinstance(arr, list) and arr and isinstance(arr[0], (list, tuple)) and len(arr[0]) == 2:
        for p in arr:
            if p[0] == height:
                p[1] = 0
                return arr
        return arr
    a = correct(arr)
    idx = height + 1
    if idx < len(a): a[idx] = 0
    return a

def comma_format(num, precision=0):
    a = correct(num)
    if len(a) == 2:
        val = a[1]
        if precision == 0: return f"{int(round(val)):,}"
        else: return f"{val:,.{precision}f}"
    try: return string(a)
    except Exception: return str(a)

def regular_format(num, precision):
    a = correct(num)
    if len(a) == 2:
        val = a[1]
        if precision == 0: return f"{int(val):,}"
        else: return f"{val:.{precision}f}"
           
    return string(a)

# End of stuff from https://github.com/cloudytheconqueror/letter-notation-format
def _is_int_like(x):
    v = tofloat(x)
    return v is not None and abs(v - round(v)) < 1e-12

# Actual stuff
# Comparison functions
def lt(a, b): return compare(a, b) == -1
def gt(a, b): return compare(a, b) == 1
def eq(a, b): return compare(a, b) == 0
def lte(a, b): return compare(a, b) != 1
def gte(a, b): return compare(a, b) != -1
def maximum(a, b):
    if gte(a,b): return correct(a)
    else: return correct(b)
def minimum(a, b):
    if lte(a,b): return correct(a)
    else: return correct(b)
# Operations
def tofloat(x):
    a = correct(x)
    if len(a) == 2: return (-a[1] if a[0] == 1 else a[1])
    return None

def _lambertw_float(r, tol=1e-12, max_iter=100):
    if not math.isfinite(r):
        raise ValueError("lambertw: non-finite input")
    if r < -0.3678794411714423:
        raise ValueError("lambertw is unimplemented for results less than -1/e on the principal branch")
    if r == 0: return 0
    if r == 1: return 0.5671432904097839
    t = 0 if r < 10 else (math.log(r) - math.log(math.log(r)))
    for _ in range(max_iter):
        n = (r * math.exp(-t) + t*t) / (t + 1)
        if abs(n - t) < tol * max(1, abs(n)): return n
        t = n
    raise RuntimeError(f"lambertw: iteration failed to converge: {r}")

def lambertw(x):
    X = correct(x)
    if lt(X, [1, 0.3678794411714423]): raise ValueError("lambertw is unimplemented for results less than -1/e on the principal branch")
    if eq(X, 0): return [0, 0]
    if eq(X, 1): return [0, 0.5671432904097839]
    r = tofloat(X)
    if r is not None: return correct(_lambertw_float(r))
    L1 = ln(X)
    L2 = ln(L1)
    approx = add(subtract(L1, L2), divide(L2, L1))
    return correct(approx)

def log(x):
    arr = correct(x)
    if arr[0] == 1: raise ValueError("Can't log a negative")
    if len(arr) == 2: return correct(_log10(arr[1]))
    if len(arr) == 3: return correct([0, arr[1], arr[2] - 1])
    if len(arr) > 3: return correct(arr)
    return correct(arr)


def slog(x): return hyper_log(x, 2)
def plog(x): return hyper_log(x, 3)
def hlog(x): return hyper_log(x, 4)
def hyper_log(x, k):
    if not _is_int_like(k) or tofloat(k) < 0: raise ValueError("hyper_log height must be a non-negative integer-like value")
    if k < 1: raise ValueError("k must be >= 1")
    arr = correct(x)
    if arr[0] == 1: raise ValueError("Can't hyper_log a negative")
    if lte(arr, 10): return correct(_log10(arr[1]))
    if k == 1: return correct(_log10(tofloat(arr)))
    if lte(arr, [0, 10000000000] + [8] * max(0, k - 2)): return correct(_log10(tofloat(hyper_log(arr, k - 1))) + 1)
    if len(arr) < (k + 1): return correct(_log10(tofloat(hyper_log(hyper_log(arr, k - 1), k - 1))) + 2)
    if len(arr) == (k + 1): return correct(tofloat(hyper_log(arr[:k], k)) + arr[k])
    if len(arr) == (k + 2): return correct([0] + arr[1:(k + 1)] + [arr[k + 1] - 1])
def addlayer(x):
    arr = correct(x)
    if arr[0] == 1 and len(arr) == 2: return correct([0, 10**(-arr[1])])
    if arr[0] == 1 and len(arr) > 2: return [0, 0]
    if len(arr) == 2: return correct([0, arr[1], 1])
    if len(arr) == 3: return correct([0, arr[1], arr[2] + 1])
    if len(arr) > 3: return arr
    return arr

def add(a, b):
    A = correct(a)
    B = correct(b)
    if (len(A) == 3 and A[2] > 1) or (len(B) == 3 and B[2] > 1): return maximum(A, B)
    if len(A) == 4 or len(B) == 4: return maximum(A, B)
    if len(A) == 2 and len(B) == 2:
        sign_a = -1 if A[0] == 1 else 1
        sign_b = -1 if B[0] == 1 else 1
        result_val = sign_a * A[1] + sign_b * B[1]
        result_sign = 0 if result_val >= 0 else 1
        return correct([result_sign, abs(result_val)])
    if len(A) >= 3 and len(B) >= 3:
        if A[0] != B[0]: return maximum(a, b)
        if A[1] > B[1]:
            diff = B[1] - A[1]
            if diff < -15: log_val = A[1]
            else: log_val = A[1] + _log10(1 + 10**diff)
        else:
            diff = A[1] - B[1]
            if diff < -15: log_val = B[1]
            else: log_val = B[1] + _log10(1 + 10**diff)
        return correct([A[0], log_val] + ([1] if len(A) > 2 else []))
    if (len(A) >= 3 and len(B) == 2) or (len(B) >= 3 and len(A) == 2):
        if len(B) >= 3:
            A, B = B, A
        if A[0] != B[0]: return maximum(A, B)
        try:
            LA = log(A)
            LB = log(B)
            fa = tofloat(LA)
            fb = tofloat(LB)
        except Exception:
            return maximum(A, B)
        if fa is not None and fb is not None:
            if fa > fb:
                diff = fb - fa
                if diff < -15: log_val = fa
                else: log_val = fa + _log10(1 + 10**diff)
            else:
                diff = fa - fb
                if diff < -15: log_val = fb
                else: log_val = fb + _log10(1 + 10**diff)
            return correct([A[0], log_val] + ([1] if len(A) > 2 else []))
        return maximum(A, B)
    return maximum(A, B)

def subtract(a, b): return add(a, neg(b))

def multiply(a, b):
    A = correct(a)
    B = correct(b)
    result_sign = A[0] ^ B[0]
    if len(A) == 2 and len(B) == 2:
        val = (A[1] if A[0] == 0 else -A[1]) * (B[1] if B[0] == 0 else -B[1])
        return correct([0 if val >= 0 else 1, abs(val)])
    result = addlayer(add(log(A), log(B)))
    return result if result_sign == 0 else neg(result)

def divide(a, b):
    A = correct(a)
    B = correct(b)
    result = subtract(log(A), log(B))
    return addlayer(result)

def power(a, b):
    A = correct(a)
    B = correct(b)
    if B[0] == 1 and A[0] != 1: return divide(1, power(a,neg(b)))
    if B[0] == 1 and A[0] == 1: return divide(1, power(neg(a),neg(b)))
    if A[0] == 1: return addlayer(multiply(log(neg(A)), B))
    return addlayer(multiply(log(A), B))

def factorial(n):
    n= correct(n)
    if n[0] == 1: raise ValueError("Can't factorial a negative")
    if lte(n, [0, 170]): return correct(str(math.gamma(n[1] + 1)).strip("+"))
    term1 = multiply(add(n, 0.5), log(n))
    term2 = neg(multiply(n, 0.4342944819032518))
    total_log = add(add(term1, term2), 0.3990899341790575)
    return addlayer(total_log)

def floor(x):
    x = correct(x)
    if len(x) == 2: return correct(str(math.floor(x[1])).strip("+"))
    else: return x

def ceil(x):
    x = correct(x)
    if len(x) == 2: return correct(str(math.ceil(x[1])).strip("+"))
    else: return x

def gamma(x):
	x = correct(x)
	return factorial(sub(x,1))

# From ExpantaNum.js
def tetration(a, r):
    a = correct(a)
    r = correct(r)
    LOOP_CAP = 20

    if lte(r, -2):
        raise ValueError("tetr(a, r): undefined for r <= -2 on the principal branch")

    if eq(a, 0):
        if eq(r, 0):
            raise ValueError("0^^0 is undefined")
        if _is_int_like(r): return correct(0 if int(tofloat(r)) % 2 == 0 else 1)
        raise ValueError("tetr(0, r) with non-integer r is not supported")

    if eq(a, 1):
        if eq(r, -1):
            raise ValueError("1^^(-1) is undefined")
        return [0, 1]
    if gt(r,[0,1,1,MAX_SAFE_INT]) or gt(a,[0, 1,1,1,MAX_SAFE_INT]): return maximum(a,r)
    if gte(r,MAX_SAFE_INT) and lte(r,[0, 1,MAX_SAFE_INT]): return add(slog(a), r) + [1]
    if gt(r,[0, 1,MAX_SAFE_INT]) or gt(a,[0, 1,1,MAX_SAFE_INT]):
        q = r[:3] + [(r[3] + 1)]
        return maximum(q,a)
    if eq(r, -1): return [0, 0]
    if eq(r, 0): return [0, 1]
    if eq(r, 1): return a
    if eq(r, 2): return power(a, a)
    if eq(a, 2):
        if eq(r, 3): return [0, 16]
        if eq(r, 4): return [0, 65536]
    if lt(a, 1.444667861009766):
        n = neg(ln(a))
        return divide(lambertw(n), n)
    s = tofloat(r)
    if s is None:
        try:
            if lt(a, 1.444667861009766):
                n = neg(ln(a))
                return divide(lambertw(n), n)
        except Exception:
            pass
        raise ValueError("tetr(a, r): r is too large for iterative evaluation in this simplified implementation")

    u = int(s)
    frac = s - u
    if frac > 1e-15: f = power(a, frac)
       
    else:
        f = a
        if u > 0: u -= 1
    last = None
    h = 0

    c = [0, 308, 1]
    if  LOOP_CAP > h:
        while u != 0 and lt(f, c) and h < LOOP_CAP:
            if u > 0:
                f_next = power(a, f)
                if last is not None and eq(f_next, last):
                    u = 0
                    break
                last = f_next
                f = f_next
                u -= 1
            else:
                f_next = logbase(f, a)
                if last is not None and eq(f_next, last):
                    u = 0
                    break
                last = f_next
                f = f_next
                u += 1
            h += 1

    if h == LOOP_CAP or lt(a, 1.444667861009766): u = 0

    f_arr = correct(f)
    if u != 0:
        if len(f_arr) == 2: f_arr = [f_arr[0], f_arr[1], u]
        else:
            while len(f_arr) < 3:
                f_arr.append(0)
            f_arr[2] = f_arr[2] + u
        f = correct(f_arr)
    else: f = correct(f_arr)
    return f
def arrow(base, arrows, n, a_arg=0):
    def _arrow_raw(base, arrows, n, a_arg=0):
        r_correct = correct(arrows)
        if not _is_int_like(arrows) or tofloat(r_correct) < 0: raise ValueError("arrows must be a non-negative integer-like value")
        r = int(tofloat(r_correct))
        t = correct(base)
        n = correct(n)
        if lt(n, [0,0]):
            raise ValueError("n must be >= 0")
        if r == 0: return multiply(t, n)
        if r == 1: return power(t, n)
        if r == 2: return tetration(t, n)
        if r > MAX_SAFE_INT: return addlayer(r_correct)
        if tofloat(n) is None:
            arr_n = correct(n)
            arr_res = arr_n[:]
            target_len = r + 2
            while len(arr_res) < target_len:
                arr_res.append(0)
            arr_res[-1] = 1
            return correct(arr_res)
        if tofloat(t) is None:
            arr_t = correct(t)
            arr_res = arr_t[:]
            target_len = r + 1
            while len(arr_res) < target_len:
                arr_res.append(0)
            s_n = tofloat(n)
            if s_n is not None and abs(s_n - round(s_n)) < 1e-12:
                val = int(round(s_n)) - 1
                if val < 0: val = 0
                arr_res[-1] = val
            else:
                arr_res[-1] = 1
            return correct(arr_res)
        thr_str_r = f"10{{{r}}}9007199254740991"
        thr_str_rp1 = f"10{{{r+1}}}9007199254740991"
        try:
            thr_r = correct(thr_str_r)
            thr_rp1 = correct(thr_str_rp1)
        except Exception:
            thr_r = [0, 10000000000, 306]
            thr_rp1 = [0, 1e309 if hasattr(math, 'inf') else 1e308]
        if gte(t, thr_r) or tofloat(n) is None and gt(n, [0, MAX_SAFE_INT]): return maximum(t, n)
        s = tofloat(n)
        if s is not None and abs(s - round(s)) < 1e-12:
            u = int(round(s))
            if u <= 0: return [0, 1]
            i = t
            if u > 0: u -= 1
            fcount = 0
            limit = thr_rp1
            while u != 0 and lt(i, limit) and fcount < 100:
                i = _arrow_raw(t, r-1, i, a_arg + 1)
                u -= 1
                fcount += 1
            if fcount == 100: return correct([[0, 10], [r, 1]])
            try:
                if len(i) >= r:
                    idx = r
                    if idx < len(i): i[idx] = i[idx] + u
                    else:
                        while len(i) <= idx: i.append(0)
                        i[idx] = i[idx] + u
                    return i
            except Exception:
                pass
            return correct(i)
        if s is not None:
            u = math.floor(s)
            frac = s - u
            if frac > 1e-15: i = _arrow_raw(t, r-1, frac, a_arg + 1)
            else:
                i = t
                if u > 0: u -= 1
            fcount = 0
            limit = thr_rp1
            while u != 0 and lt(i, limit) and fcount < 100:
                if u > 0:
                    i = _arrow_raw(t, r-1, i, a_arg + 1)
                    u -= 1
                else:
                    break
                fcount += 1
            if fcount == 100: return correct([[0, 10], [r, 1]])
            try:
                if len(i) >= r:
                    idx = r
                    if idx < len(i): i[idx] = i[idx] + u
                    else:
                        while len(i) <= idx: i.append(0)
                        i[idx] = i[idx] + u
                    return i
            except Exception:
                pass
            return correct(i)
        return maximum(t, n)
    res = _arrow_raw(base, arrows, n, a_arg) # why this? cuz the output is weird
    return correct(res)
def pentation(a,b): return arrow(a,3,b)
def hexation(a,b): return arrow(a,4,b)
def heptation(a,b): return arrow(a,5,b)
def octation(a,b): return arrow(a,6,b)
def nonation(a,b): return arrow(a,7,b)
def decation(a,b): return arrow(a,8,b)
def logbase(a,b): return divide(log(a),log(b))
def ln(a): return divide(log(a),0.4342944819032518)
def sqrt(a): return power(a, 0.5)
def root(a,b): return power(a, divide(1,b))
def exp(x): return power(2.718281828459045, x)
# Short names
def hept(a,b): return heptation(a,b)
def hex(a,b): return hexation(a,b)
def pent(a,b): return pentation(a,b)
def tetr(a,b): return tetration(a,b)
def pow(a,b): return power(a,b)
def sub(a,b): return subtract(a,b)
def div(a,b): return divide(a,b)
def mul(a,b): return multiply(a,b)
def fact(a): return factorial(a)
# Formats
def string(arr, top=True):
    arr = correct(arr)
    sign = "-" if arr[0] == 1 and top else ""
    if len(arr) == 2: return f"{sign}{arr[1]}"
    e_count = arr[2]
    if e_count <= 10: inner = f"{'e'*e_count}{arr[1]}"
    else: inner = f"(10^)^{e_count} {arr[1]}"
    for d in range(3, len(arr)):
        n = arr[d]
        if n == 0: continue
        layer_depth = d - 1
        if layer_depth <= 3:
            arrows = "^" * layer_depth
            arrow_str = f"10{arrows}"
        else: arrow_str = f"10{{{layer_depth}}}"
        if n < 2: inner = f"{arrow_str}{inner}"
        else: inner = f"({arrow_str})^{n} {inner}"
    return sign + inner

def hyper_e(x):
    arr = correct(x)
    sign = "-" if arr[0] == 1 else ""
    if len(arr) > 3:
        after = [v + 1 for v in arr[3:]]
        arr = arr[:3] + after
    return sign + "E" + "#".join(map(str, arr[1:]))
# Literally a straight copy from the roblox OmegaNum.lua
def _suffix(x):
    x = correct(x)
    if x[0] == 1: return "-" + _suffix([0] + x[1:])
    if len(x) == 3 and x[2] == 2 and x[1] < 308.2547155599167: x = [0, 10**x[1], x[2]-1]
    if len(x) > 2 and x[2] > 2 or _log10(x[1]) >= 308.2547155599167: return x
    if len(x) == 2:
        num_val = x[1]
        if num_val < 1000: return str(round(num_val, decimals))
        exponent = math.floor(_log10(num_val))
        mantissa = num_val / (10 ** exponent)
        SNumber = exponent
        SNumber1 = mantissa
    elif len(x) == 3:
        SNumber = x[1]
        SNumber1 = 1

    leftover = SNumber % 3
    SNumber = math.floor(SNumber / 3) - 1
    
    if SNumber <= -1:
        base_num = SNumber1 * (10 ** leftover)
        return str(round(base_num, decimals))
    
    if SNumber == 0:
        base_num = SNumber1 * (10 ** leftover)
        return str(round(base_num, decimals)) + "K"
    elif SNumber == 1:
        base_num = SNumber1 * (10 ** leftover)
        return str(round(base_num, decimals)) + "M"
    elif SNumber == 2:
        base_num = SNumber1 * (10 ** leftover)
        return str(round(base_num, decimals)) + "B"
    
    txt = ""
    
    def suffixpart(n):
        nonlocal txt
        Hundreds = math.floor(n / 100)
        n = n % 100
        Tens = math.floor(n / 10)
        Ones = n % 10
        txt += FirstOnes[Ones]
        txt += SecondOnes[Tens]
        txt += ThirdOnes[Hundreds]
    
    def suffixpart2(n):
        nonlocal txt
        if n > 0: n += 1
        if n > 1000: n = n % 1000
        Hundreds = math.floor(n / 100)
        n = n % 100
        Tens = math.floor(n / 10)
        Ones = n % 10
        txt += FirstOnes[Ones]
        txt += SecondOnes[Tens]
        txt += ThirdOnes[Hundreds]
    
    if SNumber < 1000:
        suffixpart(SNumber)
        base_num = SNumber1 * (10 ** leftover)
        return str(round(base_num, decimals)) + txt
    
    for i in range(len(MultOnes)-1, -1, -1):
        power_val = 10 ** (i * 3)
        if SNumber >= power_val:
            part_val = math.floor(SNumber / power_val)
            suffixpart2(part_val - 1)
            txt += MultOnes[i]
            SNumber = SNumber % power_val
    
    base_num = SNumber1 * (10 ** leftover)
    return str(round(base_num, decimals)) + txt

def suffix(x):
    x = correct(x)
    if gt(x, [0, 10000000000, 8]): return format(x)
    if lt(x, [0, max_suffix, 1]): return _suffix(x)
    max_repeats = 10
    e_count = 0
    while gte(x, [0, max_suffix, 1]) and e_count < max_repeats:
        x = log(x)
        e_count += 1
    if e_count == 0: return correct(x)
    return "e" * e_count + _suffix(x)

# From https://github.com/cloudytheconqueror/letter-notation-format
def format(num, small=False):
    precision2 = max(3, decimals)
    precision3 = max(4, decimals)
    precision4 = max(10, decimals)
    n = correct(num)
    if len(n) == 2 and abs(n[1]) < 1e-308: return f"{0:.{decimals}f}"
    if n[0] == 1: return "-" + format(neg(n), decimals)
    if lt(n, 0.0001):
        inv = 1/tofloat(n)
        return format(inv, decimals) + "⁻¹"
    elif lt(n, 1): return regular_format(n, decimals + (2 if small else 0))
    elif lt(n, 1000): return regular_format(n, decimals)
    elif lt(n, 1e9): return comma_format(n)
    elif lt(n, [0, 10000000000, 3]):
        bottom = array_search(n, 0)
        rep = array_search(n, 1) - 1
        if bottom >= 1e9:
            bottom = _log10(bottom)
            rep += 1
        m = 10 ** (bottom - math.floor(bottom))
        e = math.floor(bottom)
        p = precision2 if bottom < 1_000_000 else 0
        return ("e" * int(rep)) + regular_format([0, m], p) + "e" + comma_format(e)
    elif lt(n, [0, 10000000000, 999998]):
        pol = polarize(n)
        return regular_format([0, pol['bottom']], precision3) + "F" + comma_format(pol['top'])
    elif lt(n, [0, 10000000000, 8, 3]):
        rep = array_search(n, 2)
        if rep >= 1:
            n_arr = set_to_zero(n, 2)
            return ("F" * int(rep)) + format(n_arr, decimals)
        n_val = array_search(n, 1) + 1
        if gte(n, [0, 10, n_val]):
            n_val += 1
        return "F" + format(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 999998]):   
        pol = polarize(n)
        return regular_format([0, pol['bottom']], precision3) + "G" + comma_format(pol['top'])
    elif lt(n, [0, 10000000000, 8, 8, 3]):
        rep = array_search(n, 3)
        if rep >= 1:
            n_arr = set_to_zero(n, 3)
            return ("G" * int(rep)) + format(n_arr, decimals)
        n_val = array_search(n, 2) + 1
        if gte(n, [0, 10, 0, n_val]):
            n_val += 1
        return "G" + format(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 8, 999998]):
        pol = polarize(n)
        return regular_format([0, pol['bottom']], precision3) + "H" + comma_format(pol['top'])
    elif lt(n, [0, 10000000000, 8, 8, 8, 3]):
        rep = array_search(n, 4)
        if rep >= 1:
            n_arr = set_to_zero(n, 4)
            return ("H" * int(rep)) + format(n_arr, decimals)
        n_val = array_search(n, 3) + 1
        if gte(n, [0, 10, 0, 0, n_val]):
            n_val += 1
        return "H" + format(n_val, decimals)
    else:
        pol = polarize(n, True)
        val = _log10(pol['bottom']) + pol['top']
        return regular_format([0, val], precision4) + "J" + comma_format(pol['height'])
bot.run(token)


