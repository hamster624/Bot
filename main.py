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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import traceback
OWNER_ID = 715212498109726771 # Yup thats my id
log_channel_id = None
log_channels = {}

_executor = ThreadPoolExecutor(max_workers=4)
EVAL_TIMEOUT = 0.1
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

# ---------------------
# Bot Events
# ---------------------
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"✅ Logged in as {bot.user}")
# ---------------------
# Calculator Commands (!calc + /calc)
# ---------------------
@bot.command()
async def guide(ctx):
    global log_channel_id
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None
    if ctx.guild is None:
        if isinstance(ctx.channel, discord.PartialMessageable):
            channel_name = getattr(ctx.channel, "name", "unknown")
            location = f"#{channel_name} (hidden – bot not a member)"
        else:
            location = "DMs"
    else:
        location = f"#{ctx.channel.name} ({ctx.guild.name})"

    # -------------------------------------------------------
    # Logging
    # -------------------------------------------------------
    if log_channel:
        await log_channel.send(
            f"[!guide] {ctx.author} (ID: {ctx.author.id}) in {location}"
        )
    formats = ["format","array","hyper_e","suffix", "string"]
    operations = ["arrow (arrow)","hept (heptation)","hex (hexation)","pent (pentation)", "tetr (tetration)", "pow (power)", "exp", "root", "sqrt", "addlayer",
                  "log", "ln", "logbase", "slog", "plog (penta-log)", "hlog (hexa-log)", "hyper_log", "lambertw",
                  "fact (factorial)", "gamma",
                  "add (addition)", "sub (subtract)", "mul (multiply)", "div (division)",
                  "eq", "lt", "gt", "gte", "lte", "min", "max",
                  "floor", "ceil"]
    help_message = "**📘 !calc Help**\n\n"
    help_message += "**Available Formats:**\n" + ", ".join(formats) + "\n\n"
    help_message += "**Supported Operations:**\n" + ", ".join(operations) + "\n\n"
    help_message += "!!**DO NOT USE ^, *, /, -, + **!! Use the operations that are above this text, if not then there is a big chance of something not working correctly. Correct usage: 10^10 -> pow(10, 10) or 45+55 -> add(45, 55) or 10^10^10 -> pow(10, pow(10, 10))\n"
    help_message += "For the arrow operation the 1st number is the base, 2nd number are the arrows and the 3rd is the height so arrow(5,4,8)=5^^^^8.\n"
    help_message += "Hyper_log(num, height) lets you set the height of the number you log. Example: hyper_log(1000,3) is penta-log of 1000 because the height of 3 is pentation (^^^).\n"
    help_message += "Usage: `!calc <expression> [format]`\n"
    help_message += "Number Usage: for numbers below 2^1024 you can use float (normal numbers), but after they go higher use either the string format so '10^^10' or the hyper_e which is 'E10000000000#8' or the array format [0, 10000000000, 8] (shouldnt be in ' ') and the '1F10' format. **MAKE SURE ALL OF THESE EXCEPT THE ARRAY FORMAT ARE IN ' '**"
    await ctx.send(help_message)
def _eval_in_subprocess(expr: str, safe_globals: dict, q: mp.Queue):
    try:
        res = eval(expr, safe_globals, {})
        q.put(('OK', res))
    except Exception as e:
        q.put(('ERR', e))

def run_eval_in_subprocess(expr: str, safe_globals: dict, timeout: float):
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_eval_in_subprocess, args=(expr, safe_globals, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds (process terminated).")
    if not q.empty():
        status, payload = q.get()
        if status == 'OK':
            return payload
        else:
            raise payload

    else:
        raise RuntimeError("No output from subprocess (maybe it crashed or couldn't pickle result).")

async def safe_eval_process(expr: str, safe_globals: dict, timeout: float = EVAL_TIMEOUT):
    loop = asyncio.get_running_loop()
    start = time.time()
    try:
        result = await loop.run_in_executor(_executor, run_eval_in_subprocess, expr, safe_globals, timeout)
        elapsed = time.time() - start
        return result, elapsed
    except TimeoutError:
        raise
    except Exception:
        raise
@bot.command()
async def setlogchannel(ctx, channel: discord.TextChannel):
    """Sets the log channel for all calc logs (owner-only)."""
    global log_channel_id
    if ctx.author.id != OWNER_ID:
        await ctx.send("🚫 Only the bot owner can set the log channel.")
        return

    log_channel_id = channel.id
    await ctx.send(f"✅ Log channel set to {channel.mention} for all servers and DMs.")

@bot.command()
async def calc(ctx, *, expression: str):
    global log_channel_id
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    formats = {
        "format": format,
        "string": string,
        "hyper_e": hyper_e,
        "suffix": suffix,
        "array": correct,
    }

    try:
        # ---- Parse format ----
        tokens = expression.split()
        fmt_name = "format"

        if tokens and tokens[-1].lower() in formats:
            fmt_name = tokens[-1].lower()
            tokens = tokens[:-1]

        expr = " ".join(tokens).replace("^", "**")

        if ctx.guild is None:
            location = "Server (hidden – bot not a member)" if isinstance(ctx.channel, discord.PartialMessageable) else "DMs"
        else:
            location = f"#{ctx.channel.name} ({ctx.guild.name})"

        # ---- Logging ----
        if log_channel:
            await log_channel.send(
                f"[!calc] {ctx.author} (ID: {ctx.author.id}) "
                f"in {location} "
                f"sent: {expr} | format: {fmt_name}"
            )

        # ---- Safe eval environment ----
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

        # ---- Evaluate ----
        try:
            value, elapsed = await safe_eval_process(expr, safe_globals, timeout=EVAL_TIMEOUT)
        except TimeoutError:
            await ctx.reply("⏱ Took too long (>0.1s) — skipped.", mention_author=False)
            return

        result = formats.get(fmt_name, format)(value)

        await ctx.reply(
            f"**Result:** ```{result}```\n⏱ Evaluated in {elapsed:.6f} seconds \nUse /guide for usage.",
            mention_author=False
        )
    except SyntaxError as e:
        await interaction.followup.send(f"❌ Syntax Error: Invalid syntax. Check /guide for proper syntax.")
    except Exception as e:
        await ctx.reply(f"❌ Error: `{e}`", mention_author=False)
# ---------------------
# Guide Command (slash)
# ---------------------
@bot.tree.command(name="guide", description="Shows help for the calculator")
async def guide_slash(interaction: discord.Interaction):
    global log_channel_id
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    if interaction.guild is None:location = "Server (hidden – bot not a member)" if isinstance(interaction.channel, discord.PartialMessageable) else "DMs"
    else:location = f"#{interaction.channel.name} ({interaction.guild.name})"
    if log_channel:
        await log_channel.send(
            f"[/guide] {interaction.user} (ID: {interaction.user.id}) "
            f"in {location}"
        )
    formats = ["format","array","hyper_e","suffix", "string"]
    operations = ["arrow (arrow)","hept (heptation)","hex (hexation)","pent (pentation)", "tetr (tetration)", "pow (power)", "exp", "root", "sqrt", "addlayer",
                  "log", "ln", "logbase", "slog", "plog", "hlog", "hyper_log", "lambertw",
                  "fact (factorial)", "gamma",
                  "add (addition)", "sub (subtract)", "mul (multiply)", "div (division)",
                  "eq", "lt", "gt", "gte", "lte", "min", "max",
                  "floor", "ceil"]
    help_message = "**📘 /calc Help**\n\n"
    help_message += "**Available Formats:**\n" + ", ".join(formats) + "\n\n"
    help_message += "**Supported Operations:**\n" + ", ".join(operations) + "\n\n"
    help_message += "!!**DO NOT USE ^, *, /, -, + **!! Use the operations that are above this text, if not then there is a big chance of something not working correctly. Correct usage: 10^10 -> pow(10, 10) or 45+55 -> add(45, 55) or 10^10^10 -> pow(10, pow(10, 10))\n"
    help_message += "For the arrow operation the 1st number is the base, 2nd number are the arrows and the 3rd is the height so arrow(5,4,8)=5^^^^8.\n"
    help_message += "Hyper_log(num, height) lets you set the height of the number you log. Example: hyper_log(1000,3) is penta-log of 1000 because the height of 3 is pentation (^^^).\n"
    help_message += "Usage: `/calc <expression> [format]`\n"
    help_message += "Number Usage: for numbers below 2^1024 you can use float (normal numbers), but after they go higher use either the string format so '10^^10' or the hyper_e which is 'E10000000000#8' or the array format [0, 10000000000, 8] (shouldnt be in ' ') and the '1F10' format. **MAKE SURE ALL OF THESE EXCEPT THE ARRAY FORMAT ARE IN ' '**"
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
    global log_channel_id
    log_channel = bot.get_channel(log_channel_id) if log_channel_id else None

    if interaction.guild is None:
        if isinstance(interaction.channel, discord.PartialMessageable):
            channel_name = getattr(interaction.channel, "name", "unknown")
            location = f"#{channel_name} (hidden – bot not a member)"
        else:
            location = "DMs"
    else:
        location = f"#{interaction.channel.name} ({interaction.guild.name})"

    formats = {
        "format": format,
        "string": string,
        "hyper_e": hyper_e,
        "suffix": suffix,
        "array": correct,
    }

    fmt_name = fmt.lower()
    if fmt_name not in formats:
        fmt_name = "format"

    if log_channel:
        await log_channel.send(
            f"[/calc] {interaction.user} (ID: {interaction.user.id}) "
            f"in {location} "
            f"sent: {expression} | format: {fmt_name}"
        )

    try:
        await interaction.response.defer(thinking=True)
    except Exception:
        logging.exception("Failed to defer interaction (fallback to followup).")

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

        try:
            value, elapsed = await safe_eval_process(expr, safe_globals, timeout=EVAL_TIMEOUT)
        except TimeoutError:
            await interaction.followup.send("⏱ Took too long (>0.1s) — skipped.")
            return

        result = formats[fmt_name](value)

        await interaction.followup.send(
            f"**Result:** ```{result}```\n"
            f"⏱ Evaluated in {elapsed:.6f} seconds\n"
            f"Use /guide for the correct usage of this command."
        )

    except SyntaxError as e:
        await interaction.followup.send(f"❌ Syntax Error: Invalid syntax. Check /guide for proper syntax.")
    except Exception as e:
        await interaction.followup.send(f"❌ Error: `{e}`")

import math
# if you want to do more than 900 arrows uncomment the next 2 lines. (Note: You dont need to do this if precise_arrow = False)
#import sys
#sys.setrecursionlimit(100000)
#--Edtiable things--
decimals = 6 # How many decimals (duh). Max 16
precise_arrow = True # Makes the arrows beyond "arrow_precision" to be less precise for a large speed increase. True means it uses full precision and False makes it be less precise. (Note: This doesnt work if height is less than 2).
arrow_precision = 28 # How precise the arrows should be. I found this to be the perfect number if you use the format "format" and no more is needed. (Note: This does nothing if precise_arrow = True)
max_suffix = 63 # At how much 10^x it goes from being suffix to scientific. Example: 1e1,000 -> e1K
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
# You can ignore these, these are only to help the code.
def correct(x):
    if isinstance(x, (int, float)): return correct([0 if x >= 0 else 1, abs(x)])

    if isinstance(x, str):
        s = x.strip()
        s = s.replace("1e", "e")
        if s.startswith("E") or s.startswith("-E"): return from_hyper_e(s)
        if any(c in "}^)" for c in s): return fromstring(s)
        return fromformat(s)

    if isinstance(x, list):
        arr = x[:]
        if not arr: return [0, 0]
        if len(arr) == 1: return [0 if arr[0] >= 0 else 1, abs(arr[0])]
        if arr[0] not in (0, 1): raise ValueError(f"First element must be 0 (positive) or 1 (negative) (array:{arr})")

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
                        for j in range(2, i): arr[j] = 1
                        if i == 2: arr[2] = 1
                        else: arr[i] = 0
                        if i == len(arr) - 1: arr.append(1)
                        else: arr[i+1] += 1
                    changed = True
                    break

        for i in range(1, len(arr)):
            if isinstance(arr[i], float) and arr[i] <= MAX_SAFE_INT and arr[i].is_integer(): arr[i] = int(arr[i])

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
            num_eights = 1 if z == 2 else (z - 1)
            a1 = arr[1]
            if isinstance(a1, float) and a1.is_integer(): a1 = a1
            mid = [8] * num_eights + [a1 - 2]
            arr = arr[:2] + mid + arr[i:]
        while len(arr) > 2 and arr[-1] == 0: arr.pop(-1)
        return arr
    raise TypeError("Unsupported type for correct")
def from_hyper_e(x):
    if not x.lstrip('-').startswith('E'): raise ValueError("Not a hyper_e string")
    sign = int(x.startswith('-'))
    nums = [int(n) for n in x.lstrip('-E').replace('#', ',').split(',')]
    if len(nums) > 3: nums[2:] = [v - 1 for v in nums[2:]]
    return correct([sign] + nums)

def compare(a, b):
    A = correct(a)
    B = correct(b)
    if A[0] != B[0]: return -1 if A[0] == 1 else 1
    sign = -1 if A[0] == 1 else 1
    lenA = len(A)
    lenB = len(B)
    A_layer = lenA - 2 if lenA > 2 else 0
    B_layer = lenB - 2 if lenB > 2 else 0
    if A_layer != B_layer: return sign * (1 if A_layer > B_layer else -1)
    min_len = min(lenA, lenB)
    for i in range(1, min_len + 1):
        Ai = A[-i]
        Bi = B[-i]
        if Ai != Bi:
            return sign * (1 if Ai > Bi else -1)
    if lenA != lenB: return sign * (1 if lenA > lenB else -1)
    return 0

def neg(x):
    correct(x)
    x[0] = int(not x[0])
    return x

# Everything after this is from https://github.com/cloudytheconqueror/letter-notation-format
def _to_pair_array(arr):
    if not arr: return [[0, 0]]
    if isinstance(arr[0], (list, tuple)) and len(arr[0]) == 2: return [[int(h), float(v)] for h, v in arr]
    if not isinstance(arr, (list, tuple)): raise TypeError("polarize: unsupported array type")
    if len(arr) == 1: return [[0, float(arr[0])]]
    base_val = float(arr[1])
    pairs = [[0, base_val]]
    for i in range(2, len(arr)): pairs.append([i-1, float(arr[i])])
    return pairs

def polarize(array, smallTop=False, skip=False):  
    try: array = correct(array)
    except: pass
    if skip == False:
        if len(array) >= 16:
            result = polarize(array, smallTop=smallTop, skip=True)
            return {"bottom": result["bottom"], "top": result["top"], "height": len(array)-1}

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
                        top = pairs[elem][1] + 1
                    else: top = 1
                else: top = 1
    return {"bottom": bottom, "top": top, "height": height}

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
def tofloat(a):
    if gt(a, [0, 308.25, 1]): return None
    a = correct(a)
    val = a[1]
    if len(a) == 3: val = 10**val
    return -val if a[0] == 1 else val
# Same thing but for numbers lower than 2^52-1
def tofloat2(a):
    a = correct(a)
    if not len(a) == 2: return None
    return -a[1] if a[0] == 1 else a[1]
    
def _lambertw_float(r, tol=1e-12, max_iter=100):
    if not math.isfinite(r):
        raise ValueError("lambertw: non-finite input")
    if r < -0.3678794411714423: raise ValueError("lambertw is unimplemented for results less than -1/e on the principal branch")
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
def abs_val(x): 
    x=correct(x)
    return correct([0] + x[1:])
def add(a, b):
    a, b = correct(a), correct(b)
    if gt(a, [0, 15.95458977019, 2]) or gt(b, [0, 15.95458977019, 2]): return maximum(a,b)
    if a[0] == 1 and b[0] == 1: return neg(add(neg(a),neg(b)))
    if a[0] == 1 and b[0] == 0: return subtract(b, neg(a))
    if a[0] == 0 and b[0] == 1: return subtract(a, neg(b))
    if len(a) == 3 or len(b) == 3:
        if (len(a) > 2 and a[2] > 1) or (len(b) > 2 and b[2] > 1): return maximum(a, b)
    if len(a) == 2 and len(b) == 2: return correct([0, tofloat(a) + tofloat(b)])
    loga = tofloat(log(a))
    logb = tofloat(log(b))
    M = max(loga, logb)
    m = min(loga, logb)
    return addlayer(M + tofloat(log(1 + 10**(m - M))))

def subtract(a,b):
    a, b = correct(a), correct(b)
    if eq(a,b) and a[0] == b[0]: return [0,0]
    if eq(a,b): return neg(add(abs_val(a),abs_val(b)))
    if gt(a, [0, 15.954589770191003, 2]) or gt(b, [0, 15.954589770191003, 2]):
        if gt(b,a): return neg(b)
        if gt(a,b): return a
    if a[0] == 1 and b[0] == 1: return neg(subtract(abs_val(b), abs_val(a)))
    if a[0] == 1 and b[0] == 0: return neg(addlayer(tofloat(log(abs_val(a))) + tofloat(log(1 + tofloat(addlayer(tofloat(log(b)) - tofloat(log(abs_val(a)))))))))
    if a[0] == 0 and b[0] == 1: return add(a, abs_val(b))
    if lt(a,b):
        if a[0] == 0 and b[0] == 0: return neg(addlayer(tofloat(log(a)) + tofloat(log(abs_val(1 - tofloat(addlayer(tofloat(log(b)) - tofloat(log(a)))))))))
    if a[0] == 0 and b[0] == 0: return addlayer(tofloat(log(a)) + tofloat(log(1 - tofloat(addlayer(tofloat(log(b)) - tofloat(log(a)))))))

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

    if lte(r, -2): raise ValueError("tetr(a, r): undefined for r <= -2 on the principal branch")

    if eq(a, 0):
        if eq(r, 0): raise ValueError("0^^0 is undefined")
        if _is_int_like(r): return correct(0 if int(tofloat(r)) % 2 == 0 else 1)
        raise ValueError("tetr(0, r) with non-integer r is not supported")

    if eq(a, 1):
        if eq(r, -1): raise ValueError("1^^(-1) is undefined")
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
def _arrow(t, r, n, a_arg=0, prec=precise_arrow):
    r = tofloat2(correct(r))
    if eq(r, 0): return multiply(t, n)
    if eq(r, 1): return power(t, n)
    if eq(r, 2): return tetration(t, n)
    if eq(t,2) and eq(n,2): return [0, 4]
    if prec == False and r > arrow_precision and gte(n,2):
        arrow_amount = _arrow(t,arrow_precision,n, a_arg, True)
        if eq(n,2): return [0, 10000000000] + [8] * (r-arrow_precision) + arrow_amount[-(arrow_precision):]
        return [0, 10000000000] + [8] * (r-arrow_precision) + arrow_amount[-(arrow_precision-1):]
    s = tofloat2(n)
    if s is None:
        arr_n = correct(n)
        target_len = r + 2
        arr_res = arr_n + [0] * (target_len - len(arr_n))
        arr_res[-1] = 1
        return correct(arr_res)

    s_t = tofloat2(t)
    if s_t is None:
        arr_t = correct(t)
        target_len = r + 1
        arr_res = arr_t + [0] * (target_len - len(arr_t))
        if abs(s - round(s)) < 1e-12:
            val = max(0, int(round(s)) - 1)
            arr_res[-1] = val
        else: arr_res[-1] = 1
        return correct(arr_res)

    thr_r = [0, MAX_SAFE_INT, 1]

    if gte(t, thr_r) or (tofloat2(n) is None and gt(n, [0, MAX_SAFE_INT])): return maximum(t, n)

    if abs(s - round(s)) < 1e-12:
        u = int(round(s))
        if u <= 0: return [0, 1]
        i = t
        u -= 1
        fcount = 0
        limit = thr_r
        while u != 0 and lt(i, limit) and fcount < 100:
            i = _arrow(t, r - 1, i, a_arg + 1, True)
            u -= 1
            fcount += 1
        if fcount == 100:
            return correct([[0, 10], [r, 1]])
        try:
            if len(i) >= r:
                idx = r
                if idx < len(i): i[idx] = i[idx] + u
                else:
                    i = i + [0] * (idx - len(i) + 1)
                    i[idx] = i[idx] + u
                return i
        except Exception: pass
        return correct(i)

    u = math.floor(s)
    frac = s - u
    if frac > 1e-15: i = _arrow(t, r - 1, frac, a_arg + 1, True)
    else:
        i = t
        if u > 0: u -= 1
    fcount = 0
    limit = thr_r
    while u != 0 and lt(i, limit) and fcount < 100:
        if u > 0:
            i = _arrow(t, r - 1, i, a_arg + 1)
            u -= 1
        else: break
        fcount += 1
    try:
        if len(i) >= r:
            idx = r
            if idx < len(i):
                i[idx] = i[idx] + u
            else:
                i = i + [0] * (idx - len(i) + 1)
                i[idx] = i[idx] + u
            return i
    except Exception: pass
    return correct(i)

def arrow(base, arrows, n, a_arg=0, prec=precise_arrow):
    r_correct = correct(arrows)
    if not _is_int_like(arrows) or tofloat2(r_correct) < 0: raise ValueError("arrows must be a non-negative integer-like value")
    r = int(tofloat2(r_correct))
    t = correct(base)
    n_corr = correct(n)
    if lt(n_corr, [0, 0]): raise ValueError("n must be >= 0")

    res = _arrow(t, r, n_corr, a_arg, prec)
    return correct(res)
def pentation(a,b): return arrow(a,3,b)
def hexation(a,b): return arrow(a,4,b)
def heptation(a,b): return arrow(a,5,b)
def octation(a,b): return arrow(a,6,b)
def nonation(a,b): return arrow(a,7,b)
def decation(a,b): return arrow(a,8,b)
def logbase(a,b): return divide(log(a),log(b))
def ln(a): return divide(log(a),0.4342944819032518) # log10(a)/log10(e)
def sqrt(a): return root(a,2)
def root(a,b): 
    if lt(b,0): raise ValueError("Can't root a negative")
    if gt(b,0) and lt(b,1): return power(a,divide(1,b))
    return addlayer(divide(log(a),b))
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
    if e_count <= 7: inner = f"{'e'*e_count}{arr[1]}"
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
def _suffix(x, suffix_decimals=decimals):
    x = correct(x)
    if x[0] == 1: return "-" + _suffix([0] + x[1:])
    if len(x) == 3 and x[2] == 2 and x[1] < 308.2547155599167: x = [0, 10**x[1], x[2]-1]
    if len(x) > 2 and x[2] > 2 or _log10(x[1]) >= 308.2547155599167: return x
    if len(x) == 2:
        num_val = x[1]
        if num_val < 1000: 
            val = round(num_val, suffix_decimals)
            return str(int(val) if val == int(val) else val)
        exponent = math.floor(_log10(num_val))
        mantissa = num_val / (10 ** exponent)
        SNumber = exponent
        SNumber1 = mantissa
    elif len(x) == 3:
        SNumber = x[1]
        SNumber1 = 1

    leftover = SNumber % 3
    SNumber = math.floor(SNumber / 3) - 1

    def format_with_suffix(val, suffix):
        val_rounded = round(val, suffix_decimals)
        if val_rounded >= 1000:
            val_rounded /= 1000
            next_suffix = {"K": "M", "M": "B", "B": "T"}.get(suffix, "")
            suffix = next_suffix
        return str(int(val_rounded) if val_rounded == int(val_rounded) else val_rounded) + suffix

    base_num = SNumber1 * (10 ** leftover)

    if SNumber <= -1: return str(int(round(base_num, suffix_decimals))) if base_num == int(base_num) else str(round(base_num, suffix_decimals))
    elif SNumber == 0: return format_with_suffix(base_num, "K")
    elif SNumber == 1: return format_with_suffix(base_num, "M")
    elif SNumber == 2: return format_with_suffix(base_num, "B")

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
        return format_with_suffix(base_num, "") + txt

    for i in range(len(MultOnes)-1, -1, -1):
        power_val = 10 ** (i * 3)
        if SNumber >= power_val:
            part_val = math.floor(SNumber / power_val)
            suffixpart2(part_val - 1)
            txt += MultOnes[i]
            SNumber = SNumber % power_val
    return format_with_suffix(base_num, "") + txt

def suffix(num, small=False):
    precision2 = max(5, decimals)
    precision3 = max(4, decimals)
    precision4 = max(6, decimals)
    n = correct(num)
    if len(n) == 2 and abs(n[1]) < 1e-308: return f"{0:.{decimals}f}"
    if n[0] == 1: return "-" + suffix(neg(n), decimals)
    if lt(n, 0.0001):
        inv = 1/tofloat(n)
        return "1/" + _suffix(inv)
    elif lt(n, 1): return regular_format(n, decimals + (2 if small else 0))
    elif lt(n, 1000): return regular_format(n, decimals)
    elif lt(n, 1e9): return _suffix(n)
    elif lt(n, [0, max_suffix, 1]): return _suffix(n)
       
    elif lt(n, [0, max_suffix, 2]):
        bottom = array_search(n, 0)
        rep = array_search(n, 1) - 1
        if bottom >= 1e9:
            bottom = _log10(bottom)
            rep += 1
        m = 10 ** (bottom - math.floor(bottom))
        e = math.floor(bottom)
        p = precision2
        return regular_format([0, m], p) + "e" + _suffix(e)
    elif lt(n, [0, max_suffix, 3]):
        bottom = array_search(n, 0)
        rep = array_search(n, 1) - 1
        if bottom >= 1e9:
            bottom = _log10(bottom)
            rep += 1
        m = 10 ** (bottom - math.floor(bottom))
        e = math.floor(bottom)
        p = precision2
        return "e" + regular_format([0, m], p) + "e" + _suffix(e)
    elif lt(n, [0, 10000000000, 3]):
        bottom = array_search(n, 0)
        rep = array_search(n, 1) - 1
        if bottom >= 1e9:
            bottom = _log10(bottom)
            rep += 1
        m = 10 ** (bottom - math.floor(bottom))
        e = math.floor(bottom)
        p = precision2
        return "ee" + regular_format([0, m], p) + "e" + _suffix(e)
    pol = polarize(n)
    if lt(n, [0, 10000000000, 999998]): return regular_format([0, pol['bottom']], precision3) + "F" + _suffix(pol['top'], 0)
    elif lt(n, [0, 10000000000, 8, 3]):
        rep = array_search(n, 2)
        if rep >= 1:
            n_arr = set_to_zero(n, 2)
            return ("F" * int(rep)) + suffix(n_arr, decimals)
        n_val = array_search(n, 1) + 1
        if gte(n, [0, 10, n_val]):
            n_val += 1
        return "F" + format(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 999998]): return regular_format([0, pol['bottom']], precision3) + "G" + _suffix(pol['top'], 0)
    elif lt(n, [0, 10000000000, 8, 8, 3]):
        rep = array_search(n, 3)
        if rep >= 1:
            n_arr = set_to_zero(n, 3)
            return ("G" * int(rep)) + suffix(n_arr, decimals)
        n_val = array_search(n, 2) + 1
        if gte(n, [0, 10, 0, n_val]):
            n_val += 1
        return "G" + suffix(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 8, 999998]): return regular_format([0, pol['bottom']], precision3) + "H" + _suffix(pol['top'], 0)
    elif lt(n, [0, 10000000000, 8, 8, 8, 3]):
        rep = array_search(n, 4)
        if rep >= 1:
            n_arr = set_to_zero(n, 4)
            return ("H" * int(rep)) + suffix(n_arr, decimals)
        n_val = array_search(n, 3) + 1
        if gte(n, [0, 10, 0, 0, n_val]):
            n_val += 1
        return "H" + suffix(n_val, decimals)
    else:
        pol = polarize(n, True)
        val = _log10(pol['bottom']) + pol['top']
        return regular_format([0, val], precision4) + "J" + _suffix(pol['height'])

# From https://github.com/cloudytheconqueror/letter-notation-format
def format(num, decimals=decimals, small=False):
    precision2 = max(5, decimals)
    precision3 = max(4, decimals)
    precision4 = max(6, decimals)
    n = correct(num)
    if len(n) == 2 and abs(n[1]) < 1e-308: return f"{0:.{decimals}f}"
    if n[0] == 1: return "-" + format(neg(n), decimals)
    if lt(n, 0.0001):
        inv = 1/tofloat(n)
        return "1/" + format(inv, decimals)
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
        p = precision2 if bottom < 1_000_000 else 2
        return ("e" * int(rep)) + regular_format([0, m], p) + "e" + comma_format(e)
    pol = polarize(n)
    if lt(n, [0, 10000000000, 999998]): return regular_format([0, pol['bottom']], precision3) + "F" + comma_format(pol['top'])
    elif lt(n, [0, 10000000000, 8, 3]):
        rep = array_search(n, 2)
        if rep >= 1:
            n_arr = set_to_zero(n, 2)
            return ("F" * int(rep)) + format(n_arr, decimals)
        n_val = array_search(n, 1) + 1
        if gte(n, [0, 10, n_val]):
            n_val += 1
        return "F" + format(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 999998]): return regular_format([0, pol['bottom']], precision3) + "G" + comma_format(pol['top'])
    elif lt(n, [0, 10000000000, 8, 8, 3]):
        rep = array_search(n, 3)
        if rep >= 1:
            n_arr = set_to_zero(n, 3)
            return ("G" * int(rep)) + format(n_arr, decimals)
        n_val = array_search(n, 2) + 1
        if gte(n, [0, 10, 0, n_val]):
            n_val += 1
        return "G" + format(n_val, decimals)
    elif lt(n, [0, 10000000000, 8, 8, 999998]): return regular_format([0, pol['bottom']], precision3) + "H" + comma_format(pol['top'])
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
def count_repeating(s, target=None):
    if not s:
        return 0
    if target is None:
        target = s[0]
    
    count = 0
    for ch in s:
        if ch == target:
            count += 1
        else:
            break
    return count

def fromformat(x):
    start_array = [0, 0, 0, 0, 0, 0]
    x = x.replace(",", "")
    if x.startswith("-"): 
        start_array[0] = 1
        x = x.strip("-")

    if x.startswith("H"): 
        start_array[5] = count_repeating(x)
        x = x.strip("H")
   
    if x.startswith("G"): 
        start_array[4] = count_repeating(x)
        x = x.strip("G")

    if x.startswith("F"): 
        start_array[3] = count_repeating(x)
        x = x.strip("F")
    
    if x.startswith("e") and (x.count("e") != 1):
        start_array[2] = x.count("e")-1
        x = x.strip("e")
    if 'e' in x:
        before, after = x.split("e")
        if before == "": before = 1
        start_array[1] = math.log10(float(before)) + float(after)
        if start_array[2] == 0: start_array[2] = 1
    if 'F' in x:
        before, after = x.split("F")
        start_array[2] = int(math.log10(float(before)) + float(after))
        start_array[1] = 10 ** (math.log10(float(before)) + float(after) - start_array[2])
    # This will definitely slow it down, but i cannot find a different way to do this
    if 'G' in x:
        before, after = x.split("G")
        pentated = arrow(10, 3, float(after) + math.log10(float(before)))
        start_array[3] =+ pentated[3]
        start_array[2] =+ pentated[2]
        start_array[1] =+ pentated[1]
    
    # same here
    if 'H' in x:
        before, after = x.split("H")
        hexated = arrow(10, 4, float(after) + math.log10(float(before)))
        start_array[4] =+ hexated[4]
        start_array[3] =+ hexated[3]
        start_array[2] =+ hexated[2]
        start_array[1] =+ hexated[1]
    if 'J' in x:
        before, after = x.split("J")
        return arrow(10,float(after)+1,float(before), prec=False)
    try: start_array[1] += float(x)
    except: pass
    return correct(start_array)
# Sniffed breaking bad money making stuff a bit too much to code and in the result got this code. Oh and spent 2h 15min for this trash
def fromstring(x):
    if x.startswith("(10"):
        size = x.strip("(10")
        size = count_repeating(size)
    if x.startswith("(10{"):
        size = x.strip("(10{")
        after = x.split("})^", 1)[1]
        size = int(after.split(None, 1)[0])
    if x.startswith("10{"):
        size = x.strip("10").strip("{")
        before, after = size.split("}", 1)
        size = int(before)
    if x.startswith("10^"):
        size = x.strip("10")
        size = count_repeating(size)
    if x.startswith("e"): size = 1
    array = [0] * (size+2)
    if x.startswith("-"): 
        array[0] = 1
        x = x.strip("-")
    def logic(x):
        try:  array[1] = float(x)
        except: pass
        if x.startswith("(10^"):
            x2 = x.strip("(10")
            count = count_repeating(x2)
            after = x.split(")^", 1)[1]
            num = after.split(None, 1)[0]
            array[count + 1] = int(num)
            before, after = x.split(" ", 1)
            x = after
            logic(x)
        if x.startswith("10^"):
           lst = list(x)
           x =  x.strip("10")
           count = count_repeating(x)
           array[count +1] = 1
           x= "".join(lst[2+count:])
           logic(x)
        if x.startswith("(10{"):
            x = x.strip("(10{")
            before, after = x.split("})^", 1)
            num = after.split(None, 1)[0]
            array[int(before)+1] = int(num)
            before, after = x.split(" ", 1)
            x = after
            logic(x)
        if x.startswith("10{"):
            x = x.removeprefix("10{")
            before, after = x.split("}", 1)
            array[int(before) + 1] = 1
            x = after
            logic(x)
        if x.startswith("e"):
            e = count_repeating(x)
            array[2] = e
            array[1] = float(x[x.rfind('e')+1:])
    logic(x)
    return correct(array)
bot.run(token)

