import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import time
import json
from flask import Flask
from threading import Thread

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
# Bot Setup
# ---------------------

load_dotenv()

token = os.getenv('DISCORD_TOKEN')
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
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
    """Save the current counting state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)



def load_state():
    """Load the counting state from disk."""
    global state
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {}



def get_guild_state(guild_id):
    """Get the state for a specific guild, or default if missing."""
    return state.get(str(guild_id), {
        "counting_channel_id": None,
        "current_count": 0,
        "last_counter": None
    })



def set_guild_state(guild_id, counting_channel_id, current_count, last_counter):
    """Update the guild state and save to file."""
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
    print(f"✅ Logged in as {bot.user}")

@bot.command()
@commands.has_permissions(administrator=True)
async def setcount(ctx, number: int):
    """Set the current count manually for this server."""
    guild_id = ctx.guild.id
    guild_state = get_guild_state(guild_id)
    if guild_state["counting_channel_id"] is None:
        await ctx.send("⚠ No counting channel set! Use `!setcounting #channel` first.")
        return
    set_guild_state(
        guild_id,
        guild_state["counting_channel_id"],
        number,
        guild_state["last_counter"]
    )
    await ctx.send(f"✅ Current count for {ctx.guild.name} has been set to **{number}**.")

@setcount.error

async def setcount_error(ctx, error):

    if isinstance(error, commands.MissingPermissions):

        await ctx.send("❌ You need to be an **Administrator** to set the count!")

    elif isinstance(error, commands.BadArgument):

        await ctx.send("❌ Please provide a valid integer. Example: `!setcount 42`")

@bot.command()

@commands.has_permissions(administrator=True)

async def setcounting(ctx, channel: discord.TextChannel):

    """Set the channel for counting and reset the count."""

    guild_id = ctx.guild.id

    set_guild_state(guild_id, channel.id, 0, None)

    await ctx.send(f"✅ Counting channel set to {channel.mention}. Counter reset to 0.")



@setcounting.error

async def setcounting_error(ctx, error):

    if isinstance(error, commands.MissingPermissions):

        await ctx.send("❌ You need to be an **Administrator** to set the counting channel!")



@bot.command()

async def current(ctx):

    """Show the current count for this server."""

    guild_state = get_guild_state(ctx.guild.id)

    await ctx.send(

        f"🔹 Current count: **{guild_state['current_count']}**\n"

        f"📢 Channel: <#{guild_state['counting_channel_id']}>"

        if guild_state['counting_channel_id'] else "⚠ No counting channel set!"

    )

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

        if not content:

            return



        first_token = content.split()[0]

        safe_globals = {

            "__builtins__": {},

            "math": math,
            "mp": mp,
            "tetration": tetration,

            "tetr": tetration,

            "fact": fact,

            "factorial": factorial,

            "gamma": gamma,

            "slog": slog,

            "addlayer": addlayer,

            "add": add,

            "addition": addition,

            "sub": sub,

            "subtract": subtract,

            "mul": mul,

            "multiply": multiply,

            "div": div,

            "division": division,

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

            "logbase": logbase,

            "log": log

        }

        try:

            value = eval(first_token, safe_globals, {})

            value = round(float(value))

        except Exception:

            # If the expression is invalid or unsupported, ignore the message

            return



        if message.author.id == last_counter:

            await message.channel.send(

                f"❌ {message.author.mention}, you counted twice in a row and lost! The count resets to 1."

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

@bot.command()

async def guide(ctx):

    """

    Displays all available operations and notations for the !calc command.

    """

    formats = [

        "format",

        "power10_tower",

        "correct",

        "hyper_e",

        "letter",

        "suffix_to_scientific"

    ]



    operations = [

        "tetration (tetr)", "pow (power)", "exp", "root", "sqrt", "addlayer",

        "log", "ln", "logbase", "slog", "lambertw",

        "fact (factorial)", "gamma", "OoMs",

        "add (addition)", "sub (subtract)", "mul (multiply)", "div (division)",

        "eq", "lt", "gt", "gte", "lte", "min", "max",

        "floor", "ceil"

     ]





    help_message = "**📘 !calc Help**\n\n"

    help_message += "**Available Formats:**\n" + ", ".join(formats) + "\n\n"

    help_message += "**Supported Operations:**\n" + ", ".join(operations) + "\n\n"

    help_message += "Usage: `!calc <expression> [format]`\nExample: `!calc tetr(10,10) power10_tower`"



    await ctx.send(help_message)



@bot.command()

async def calc(ctx, *, expression: str):

    """

    Evaluate an expression.

    """

    formats = {

        "format": format,

        "power10_tower": power10_tower,

        "correct": correct,

        "hyper_e": hyper_e,

        "letter": letter,

        "suffix_to_scientific": suffix_to_scientific,

    }



    try:

        tokens = expression.strip().split(" ")

        fmt_name = "format"

        if tokens[-1].lower() in formats:

            fmt_name = tokens[-1].lower()

            tokens = tokens[:-1]



        expr = " ".join(tokens)

        safe_globals = {

            "__builtins__": {},

            "math": math,
            "mp": mp,
            "tetration": tetration,

            "tetr": tetration,

            "fact": fact,

            "factorial": factorial,

            "gamma": gamma,

            "slog": slog,

            "addlayer": addlayer,

            "add": add,

            "addition": addition,

            "sub": sub,

            "subtract": subtract,

            "mul": mul,

            "multiply": multiply,

            "div": div,

            "division": division,

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

            "logbase": logbase,

            "log": log,
            "ooms": ooms
        }



        start_time = time.time()



        try:

            value = eval(expr, safe_globals, {})

        except:

            value = expr



        result = formats[fmt_name](value)



        end_time = time.time()

        elapsed = end_time - start_time



        await ctx.reply(

            f"**Result:** ```{result}```\n⏱ Evaluated in {elapsed:.6f} seconds",

            mention_author=False

        )



    except Exception as e:

        await ctx.reply(f"Error: `{e}`", mention_author=False)
import math
import mpmath as mp
# --Editable constants--
precision = 50 # amount of decimals for calculations.
format_decimals = 6 # amount of decimals for formatting output
FORMAT_THRESHOLD = 7  # the amount of e's when switching from scientific to (10^)^x format
max_layer = 10  # amount of 10^ in power10_tower format when it switches from 10^ iterated times to 10^^x
suffix_max= 1e308 # For the suffix format at how much of 10^x it adds scientific notation (max is 1e308)
# --End of editable constants--

# --Editable suffix format--
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
# --End of editable suffix format--

LARGE_HEIGHT_THRESHOLD = 9007199254740991  # 2**53-1, the largest integer that can be represented exactly in python's float
PRECISION_LIMIT = 1e-308
MIN_EXPONENT = -1e308
MAX_EXPONENT = 1e308

mp.mp.dps = precision
def get_sign_and_abs(x):
    if x is None:
        return 1, None
    if isinstance(x, (int, float, mp.mpf)):
        x_m = mp.mpf(x)
        if x_m < 0:
            return -1, -x_m
        else:
            return 1, x_m
    elif isinstance(x, str):
        if x.startswith('-'):
            return -1, x[1:]
        else:
            return 1, x
    else:
        return 1, x

def apply_sign(x, sign):
    if sign == 1:
        return x
    else:
        return negate(x)
def mpf_lim(val, max_exp=308):
    x = mp.mpf(val)
    if abs(mp.log10(abs(x))) > max_exp:
        raise OverflowError("Value exceeds max exponent limit")
    return x
def negate(x):
    if x is None:
        return None
    if isinstance(x, (int, float, mp.mpf)):
        x_m = mp.mpf(x)
        if x_m == 0:
            return mp.mpf(0)
        return -x_m
    elif isinstance(x, str):
        if x.startswith('-'):
            return x[1:]
        else:
            return '-' + x
    else:
        return '-' + str(x)

def compare_positive(a, b):
    if eq(a, b) == True:
        return 0
    elif gt(a, b) == True:
        return 1
    else:
        return -1

def tetration(a, h):
    a,h = correct2(a), correct2(h)
    try:
        h_m = mpf_lim(h)
    except:
        return "Error: Height must be a a valid number under 1e308"
    sign_a, abs_a = get_sign_and_abs(a)
    if sign_a == -1:
        return "Error: Tetration base must be non-negative"
    
    a_val = abs_a
    
    if h_m < 0:
        return "Error: Tetration height must be non-negative"
    
    try:
        a_m = mpf_lim(a_val)
        use_float = True
    except:
        use_float = False
        
    if not use_float:
        if isinstance(a_val, str):
            s = slog(a_val)
            if mp.isnan(s) or mp.isinf(s) or s == "Error: x can't be a negative number":
                return "NaN"
            try:
                return tetration(10, s + h_m - 1)
            except:
                return "NaN"
        else:
            return "NaN"
    
    a_m = mpf_lim(a_val)
    if a_m < 0:
        return "Error: Tetration base must be non-negative"
    if a_m == 0:
        if h_m == 0:
            return "NaN"
        if mp.floor(h_m) == h_m:
            if int(mp.floor(h_m)) % 2 == 0:
                return "0"
            else:
                return "1" if h_m == 1 else "0"
        else:
            return "0"
    if a_m == 1:
        return "1"

    try:
        if h_m >= mp.mpf(LARGE_HEIGHT_THRESHOLD):
            if abs(h_m - mp.nint(h_m)) < mp.mpf('1e-12'):
                height_str = format_int_scientific(int(mp.nint(h_m)))
            else:
                height_str = format_float_scientific(h_m)
            return f"10^^{height_str}"
    except NameError:
        pass
    
    log10a = mp.log10(a_m) if a_m > 0 else -mp.inf
    log_log10a = mp.log10(log10a) if log10a > 0 else -mp.inf
    
    try:
        n = mp.floor(h_m)
    except (ValueError, TypeError, OverflowError):
        return "NaN"
    
    f = h_m - n
    current = mp.power(a_m, f) if f > 0 else mp.mpf(1)
    layer = 0
    if n == 0:
        if current < mp.mpf('1e12'):
            return current
        if abs(current - mp.nint(current)) < mp.mpf('1e-10'):
            return format_float_scientific(mp.nint(current))
        return mp.nstr(current, 15)
    
    n_remaining = int(n)
    layer0_iter = 0
    prev_current = current
    while n_remaining > 0:
        if layer == 0:
            if layer0_iter >= 10000:
                if abs(current - prev_current) < mp.mpf('1e-10'):
                    break
                prev_current = current
                layer0_iter = 0
            next_log10 = current * log10a
            if next_log10 > mp.mpf('307'):
                current = next_log10
                layer = 1
            else:
                try:
                    current = mp.power(a_m, current)
                except:
                    current = next_log10
                    layer = 1
            layer0_iter += 1
            n_remaining -= 1
        elif layer == 1:
            current = log_log10a + current
            layer += n_remaining
            n_remaining = 0
        else:
            layer += n_remaining
            n_remaining = 0
    
    if layer >= 1 and mp.isfinite(current) and current > mp.mpf(LARGE_HEIGHT_THRESHOLD):
        while current > mp.mpf(LARGE_HEIGHT_THRESHOLD):
            current = mp.log10(current)
            layer += 1
    
    if layer == 0:
        if current < mp.mpf('1e12'):
            return current
        if mp.isnan(current):
            return "NaN"
        if abs(current - mp.nint(current)) < mp.mpf('1e-10'):
            return format_float_scientific(mp.nint(current))
        return f"{mp.nstr(current, n=mp.mp.dps)}"
    elif layer == 1:
        return f"e{mp.nstr(current, n=mp.mp.dps)}"
    elif layer <= FORMAT_THRESHOLD:
        return 'e' * int(layer) + f"{mp.nstr(current, n=mp.mp.dps)}"
    else:
        return f"(10^)^{layer} {mp.nstr(current, n=mp.mp.dps)}"

def slog_numeric(x, base):
    try:
        base_m = mpf_lim(base)
    except:
        return mp.nan
    if base_m <= 0 or base_m == 1:
        return mp.nan
    sign_x, abs_x = get_sign_and_abs(x)
    if sign_x == -1:
        return mp.nan
    try:
        x_val = mp.mpf(abs_x)
    except:
        return mp.nan
    
    if x_val <= 0:
        return -mp.inf
    
    count = mp.mpf('0')
    current = x_val
    while current < 1:
        if current <= 0:
            return -mp.inf
        try:
            current = mp.power(base_m, current)
        except OverflowError:
            current = mp.inf
        count -= 1
    while current > base_m:
        try:
            current = mp.log(current, base_m)
        except (OverflowError, ValueError):
            return mp.nan
        count += 1
    
    try:
        frac = mp.log(current, base_m)
    except (OverflowError, ValueError):
        return mp.nan
    return count + frac

def slog(x, base=10):
    x = correct2(x)
    sign_x, abs_x = get_sign_and_abs(x)
    if sign_x == -1:
        return "Error: x can't be a negative number"
    x = abs_x
    
    if x == 0:
        return -1
    
    if isinstance(x, str):
        if base == 10:
            if x.startswith("10^^"):
                try:
                    return mp.mpf(x[4:])
                except:
                    return mp.nan
            elif x.startswith("(10^)^"):
                parts = x.split(' ', 1)
                if len(parts) < 2:
                    return "NaN"
                head, mantissa_str = parts
                k_str = head[6:]
                try:
                    k = int(k_str)
                    mantissa = mp.mpf(mantissa_str)
                except:
                    return mp.nan
                return k + slog_numeric(mantissa, 10)
            else:
                count = 0
                s = x
                while s.startswith('e'):
                    count += 1
                    s = s[1:]
                if count == 0:
                    try:
                        return slog_numeric(mp.mpf(x), 10)
                    except:
                        return mp.nan
                else:
                    try:
                        mantissa = mp.mpf(s)
                    except:
                        return mp.nan
                    return count + slog_numeric(mantissa, 10)
        else:
            count = mp.mpf('0')
            s = x
            while s:
                if s.startswith("10^^"):
                    height_str = s[4:]
                    try:
                        height = mp.mpf(height_str)
                    except:
                        try:
                            return count + slog_numeric(mp.mpf(s), base)
                        except:
                            return mp.nan
                    count += height
                    return count
                elif s.startswith("(10^)^"):
                    parts = s.split(' ', 1)
                    if len(parts) < 2:
                        try:
                            return count + slog_numeric(mp.mpf(s), base)
                        except:
                            return mp.nan
                    head, mantissa_str = parts
                    k_str = head[6:]
                    try:
                        k = int(k_str)
                        mantissa = mp.mpf(mantissa_str)
                    except:
                        return mp.nan
                    count += k
                    s = mantissa_str
                elif s.startswith('e'):
                    idx = 0
                    while idx < len(s) and s[idx] == 'e':
                        idx += 1
                    rest = s[idx:]
                    count += idx
                    s = rest
                else:
                    try:
                        return count + slog_numeric(mp.mpf(s), base)
                    except:
                        return mp.nan
            return count
    else:
        return slog_numeric(x, base)

def log(x):
    x = correct(x)
    sign_x, abs_x = get_sign_and_abs(x)
    if sign_x == -1:
        return "Error: Logarithm of negative number"
    x = abs_x
    
    if isinstance(x, str):
        if x == "NaN" or x.startswith("Error:"):
            return x
        if x.startswith("10^^"):
            try:
                inner = mp.mpf(x[4:])
                if inner < mp.mpf(LARGE_HEIGHT_THRESHOLD):
                    try:
                        return correct("10^^" + str(inner - 1))
                    except:
                        return str(x)
            except:
                return str(x)
        elif x.startswith("(10^)^"):
            parts = x.split(' ', 1)
            if len(parts) < 2:
                return "NaN"
            head, mantissa_str = parts
            k_str = head[6:]
            try:
                k = int(k_str)
                mantissa = mp.mpf(mantissa_str)
            except:
                return "NaN"
            if k == 1:
                return str(mantissa)
            if k >= mp.mpf(LARGE_HEIGHT_THRESHOLD):
                return correct(x)
            else:
                return f"(10^)^{k-1} {mantissa_str}"
        elif x.startswith('e'):
            count = 0
            s = x
            while s.startswith('e'):
                count += 1
                s = s[1:]
            if count == 1:
                return s
            else:
                return correct('e' * (count - 1) + s)
        else:
            try:
                num_val = mp.mpf(x)
                return str(mp.log10(num_val))
            except:
                return "NaN"
    else:
        try:
            return mp.log10(x)
        except:
            return "NaN"

def logbase(a,b):
    a, b = correct(a), correct(b)
    return div(log(a),log(b))

def ln(x):
    x = correct(x)
    return mul(log(x), mp.mpf(mp.log(10)))

def addlayer(a, b=1):
    a, b = correct(a), correct(b)
    s = slog(a)
    try:
        if mp.isinf(s) or mp.isnan(s) or isinstance(s, str):
            return "NaN"
    except:
        return "NaN"
    try:
        return tetration(10, mp.mpf(b) + mp.mpf(s))
    except:
        return "Error trying to do addlayer"

def is_float_convertible(x):
    try:
        mp.mpf(x)
        return True
    except:
        return False

def subtract_positive(a, b, depth=0):
    a, b = correct(a), correct(b)
    MAX_DEPTH = 3
    if depth > MAX_DEPTH:
        return a
    if a in [0, "0"]:
        return negate(b)
    if b in [0, "0"]:
        return a
    if is_float_convertible(a) and is_float_convertible(b):
        a_float = mp.mpf(a)
        b_float = mp.mpf(b)
        result = a_float - b_float
        if result < 0:
            return negate(str(abs(result)))
        if result == 0:
            return 0
        if abs(result) < mp.mpf('1e-3') or abs(result) >= mp.mpf('1e12'):
            return format_float_scientific(result)
        return str(result)
    
    if lt(a, b) == True:
        return negate(subtract_positive(b, a, depth+1))
    
    if eq(a, b) == True:
        return 0
    
    if isinstance(a, str) and a.startswith('e') and is_float_convertible(b):
        try:
            exponent = mp.mpf(a[1:])
            if exponent > mp.mpf('1e2'):
                a_val = mp.power(10, exponent)
                if a_val > mp.mpf('1e100'):
                    return a
            else:
                a_val = mp.power(10, exponent)
            result = a_val - mp.mpf(b)
            if result <= 0:
                return 0
            if result < mp.mpf('1e12'):
                return result
            return "e" + str(mp.log10(result))
        except:
            pass 
    A = log(a)
    B = log(b)
    if A == "NaN" or B == "NaN" or A == "Error: Logarithm of negative number" or B == "Error: Logarithm of negative number":
        return a
    
    D = subtract_positive(A, B, depth+1)
    if D == "NaN" or D == "Error: Logarithm of negative number":
        return a
    
    try:
        D_float = mp.mpf(D)
        if D_float > mp.mpf('1000'):
            return a
        C = mp.power(10, D_float) - 1
        if C <= 0:
            return 0
        log10C = mp.log10(C)
        B_float = mp.mpf(B)
        new_exp = B_float + log10C
        return "e" + str(new_exp)
    except:
        return a

def add_positive(a, b):
    a, b = correct(a), correct(b)
    if a in [0, "0"]:
        return b
    if b in [0, "0"]:
        return a
    if is_float_convertible(a) and is_float_convertible(b):
        a_float = mp.mpf(a)
        b_float = mp.mpf(b)
        result = a_float + b_float
        if abs(result) < mp.mpf('1e308'):
            return result
        elif abs(result) >= mp.mpf('1e308'):
            return format_float_scientific(result)
    
    s_a = slog(a)
    s_b = slog(b)
    if mp.isnan(s_a) or mp.isnan(s_b) or isinstance(s_a, str) or isinstance(s_b, str):
        return "NaN"
    if abs(s_a - s_b) >= 1:
        return a if s_a > s_b else b
    if s_a > 100 or s_b > 100:
        return a if s_a >= s_b else b
    if s_a < 1 and s_b < 1:
        try:
            return mp.mpf(a) + mp.mpf(b)
        except:
            return a if s_a >= s_b else b
    if gt(b, a) == True:
        a, b = b, a
        s_a, s_b = s_b, s_a
    
    log_a = log(a)
    log_b = log(b)
    try:
        if is_float_convertible(log_a) and is_float_convertible(log_b):
            d_val = mp.mpf(log_b) - mp.mpf(log_a)
        else:
            d_exp = subtract(log_b, log_a)
            d_val = mp.mpf(d_exp) if is_float_convertible(d_exp) else -mp.inf
    except:
        d_val = -mp.inf
    
    if d_val < mp.mpf(MIN_EXPONENT):
        return a
    
    try:
        x = mp.power(10, d_val)
        y = mp.log10(1 + x)
    except:
        return a
    
    if is_float_convertible(log_a):
        new_exponent = mp.mpf(log_a) + y
    else:
        try:
            new_exponent = addition(log_a, y)
        except:
            return a
    
    return addlayer(new_exponent)

def addition(a, b):
    a, b = correct(a), correct(b)
    try:
        a_float = mp.mpf(a)
        b_float = mp.mpf(b)
        if abs(a_float) < mp.mpf('5e307') and abs(b_float) < mp.mpf('5e307'):
            return a_float + b_float
    except:
        pass
    
    sign_a, abs_a = get_sign_and_abs(a)
    sign_b, abs_b = get_sign_and_abs(b)
    
    if abs_a in [0, "0"] and abs_b in [0, "0"]:
        return 0
    if abs_a in [0, "0"]:
        return apply_sign(abs_b, sign_b)
    if abs_b in [0, "0"]:
        return apply_sign(abs_a, sign_a)
    
    if sign_a == sign_b:
        result = add_positive(abs_a, abs_b)
        return apply_sign(result, sign_a)
    
    cmp = compare_positive(abs_a, abs_b)
    if cmp == 0:
        return 0
    elif cmp > 0:
        result = subtract_positive(abs_a, abs_b, 0)
        return apply_sign(result, sign_a)
    else:
        result = subtract_positive(abs_b, abs_a, 0)
        return apply_sign(result, sign_b)

def subtract(a, b):
    a, b = correct(a), correct(b)
    return addition(a, negate(b))

def multiply(a, b):
    a, b = correct(a), correct(b)
    sign_a, abs_a = get_sign_and_abs(a)
    sign_b, abs_b = get_sign_and_abs(b)
    sign = sign_a * sign_b
    
    if abs_a in [0, "0"] or abs_b in [0, "0"]:
        return 0

    try:
        a_float = mp.mpf(abs_a)
        b_float = mp.mpf(abs_b)
        product = a_float * b_float
        if not mp.isinf(product) and abs(product) < mp.mpf('1e308'):
            return apply_sign(product, sign)
    except (ValueError, TypeError, OverflowError):
        pass

    try:
        log_a = log(abs_a)
        log_b = log(abs_b)
        if log_a == "Error: Logarithm of negative number" or log_b == "Error: Logarithm of negative number":
            return "Error: Logarithm of negative number"
        
        log_product = addition(log_a, log_b)
        product = addlayer(log_product)
        return apply_sign(product, sign)
    except:
        return "Error doing multiplication"

def division(a, b):
    a, b = correct(a), correct(b)
    sign_a, abs_a = get_sign_and_abs(a)
    sign_b, abs_b = get_sign_and_abs(b)
    sign = sign_a * sign_b
    
    if abs_b in [0, "0"]:
        return "Error: Division by zero"
    if abs_a in [0, "0"]:
        return 0

    try:
        a_float = mp.mpf(abs_a)
        b_float = mp.mpf(abs_b)
        quotient = a_float / b_float
        if not mp.isinf(quotient) and abs(quotient) < mp.mpf('1e308'):
            return apply_sign(quotient, sign)
    except (ValueError, TypeError, OverflowError):
        pass

    try:
        log_a = log(abs_a)
        log_b = log(abs_b)
        if log_a == "Error: Logarithm of negative number" or log_b == "Error: Logarithm of negative number":
            return "Error: Logarithm of negative number"
        
        log_quotient = subtract(log_a, log_b)
        quotient = addlayer(log_quotient)
        return apply_sign(quotient, sign)
    except:
        return "Error doing division"

def power(a, b):
    a, b = correct(a), correct(b)
    sign_a, abs_a = get_sign_and_abs(a)

    if sign_a == -1:
        try:
            b_float = mp.mpf(b)
            if abs(b_float - mp.nint(b_float)) < mp.mpf('1e-10'):
                exponent_int = int(mp.nint(b_float))
                sign_result = -1 if exponent_int % 2 == 1 else 1
                abs_result = power(abs_a, b)
                return apply_sign(abs_result, sign_result)
            else:
                return "Error: Fractional exponent of negative base"
        except:
            return "Error: Invalid exponent for negative base"

    try:
        log_a = log(abs_a)
        if log_a == "Error: Logarithm of negative number":
            return "Error: Logarithm of negative number"
        
        log_power = multiply(log_a, b)
        result = addlayer(log_power)
        return result
    except:
        return "Error doing power"
 
def exp(x):
    x = correct(x)
    return pow(mp.mpf(mp.e), x)

def root(a, b):
    a, b = correct(a), correct(b)
    if b == 0:
        return "Error: Root of order 0"
    return power(a, division(1, b))

def sqrt(x):
    x = correct(x)
    return root(x, 2)

def factorial(n):
    n = correct(n)
    sign, abs_n = get_sign_and_abs(n)
    if sign == -1:
        return "Factorial can't be negative"
    
    try:
        n_val = mp.mpf(abs_n)
    except (TypeError, OverflowError, ValueError):
        n_val = str(abs_n)
    
    if n_val == 0:
        return 1

    try:
        if n_val < 170:
            return mp.gamma(n_val + 1)
    except (ValueError, TypeError, OverflowError):
        pass
    if gt(n_val, "e1000000000000") == True:
        return addlayer(n_val)
    else:
        term1 = multiply(addition(n_val, 0.5), log(n_val))
        term2 = negate(multiply(n_val, mp.mpf('0.4342944819032518')))
        total_log = addition(addition(term1, term2), mp.mpf('0.3990899341790575'))
        return addlayer(total_log)
 
def gamma(x):
    x = correct(x)
    return fact(sub(x,1))

def floor(x):
    x = correct(x)
    try:
        return mp.floor(x)
    except:
        return x

def ceil(x):
    x = correct(x)
    try:
        return mp.ceil(x)
    except:
        return x

def lambertw(z):
    z = correct(z)
    if lte(z, 0):
        raise ValueError("Asymptotic expansion valid only for positive z >> 1")
    elif gte(z, "ee6"):
        return mul(log(z), mp.mpf('2.302585092994046'))

    L1 = ln(z)
    L2 = ln(L1)

    termC = div(L2, L1)
    numeratorD = mul(L2, sub(-2, L2))
    denominatorD = mul(2, mul(L1, L1))
    termD = div(numeratorD, denominatorD)

    part1 = sub(L1, L2)
    part2 = add(termC, termD)

    return add(part1, part2)

def ooms(start, end, time=1):
    if gt(start, end): 
        raise ValueError("OoMs error: start for the OoMs cant be more than the end")
    slg_end = slog(end)
    slg_start = slog(start)
    slg_fl_start = mp.floor(slg_start)
    slg_fl_end = mp.floor(slg_end)
    x = (tetr(10, slg_end-(slg_fl_end-1)) - tetr(10, slg_start-(slg_fl_start-1))) / mp.mpf(time)
    if x < 1 and slg_fl_end-2 < 0:
        y = slg_fl_end-2
        x = round(mp.power(10, x), 6)
    else:
        y = slg_fl_end-1
        x = round(x, 6)
    return f"{x} OoMs^{y}"

# Comparisons
def gt(a, b):
    a, b = correct(a), correct(b)
    sign_a, abs_a = get_sign_and_abs(a)
    sign_b, abs_b = get_sign_and_abs(b)
    
    if sign_a != sign_b:
        return sign_a > sign_b
    
    if sign_a == 1:
        a_slog = slog(abs_a)
        b_slog = slog(abs_b)
        try:
            if mp.isnan(a_slog) or mp.isnan(b_slog) or isinstance(a_slog, str) or isinstance(b_slog, str):
                return False
        except:
            return False
        
        if a_slog > b_slog:
            return True
        elif a_slog < b_slog:
            return False
        else:
            if is_float_convertible(abs_a) and is_float_convertible(abs_b):
                return mp.mpf(abs_a) > mp.mpf(abs_b)
            else:
                return False
    else:
        a_slog = slog(abs_a)
        b_slog = slog(abs_b)
        if mp.isnan(a_slog) or mp.isnan(b_slog) or isinstance(a_slog, str) or isinstance(b_slog, str):
            return False
        if a_slog < b_slog:
            return True
        elif a_slog > b_slog:
            return False
        else:
            if is_float_convertible(abs_a) and is_float_convertible(abs_b):
                return mp.mpf(abs_a) < mp.mpf(abs_b)
            else:
                return False

def lt(a, b):
    a, b = correct(a), correct(b)
    return gt(b, a)

def eq(a, b):
    a, b = correct(a), correct(b)
    sign_a, abs_a = get_sign_and_abs(a)
    sign_b, abs_b = get_sign_and_abs(b)
    
    if sign_a != sign_b:
        return False
    
    try:
        a_slog = slog(abs_a)
        b_slog = slog(abs_b)
    except:
        return False
    
    try:
        if mp.isnan(a_slog) or mp.isnan(b_slog) or isinstance(a_slog, str) or isinstance(b_slog, str):
            return False
    except:
        return False
    
    if abs(a_slog - b_slog) > mp.mpf('1e-10'):
        return False
    
    if is_float_convertible(abs_a) and is_float_convertible(abs_b):
        return abs(mp.mpf(abs_a) - mp.mpf(abs_b)) < mp.mpf('1e-10')
    return True

def gte(a, b):
    a, b = correct(a), correct(b)
    return not lt(a, b)

def lte(a, b):
    a, b = correct(a), correct(b)
    return not gt(a, b)

def max(a,b):
    a, b = correct(a), correct(b)
    if gte(a,b):
        return a
    else:
        return b

def min(a,b):
    a, b = correct(a), correct(b)
    if lte(a,b):
        return a
    else:
        return b
# Short names
def fact(x): return factorial(x)
def pow(a, b): return power(a, b)
def tetr(a, h): return tetration(a, h)
def mul(a, b): return multiply(a, b)
def add(a, b): return addition(a, b)
def sub(a, b): return subtract(a, b)
def div(a, b): return division(a, b)

# Formats
def hyper_e(tet, decimals=format_decimals):
    tet = correct(tet)
    if isinstance(tet, (int, float, mp.mpf)):
        return comma_format(tet, decimals)
    tet_str = str(tet)
    if tet_str.startswith("10^^"):
        height = tet_str[4:]
        return f"E10#{height}"
    if tet_str.startswith("(10^)^"):
        parts = tet_str.split(' ', 1)
        if len(parts) == 2:
            head, mant = parts
            try:
                layers = int(head[6:])
                return f"E{mant}#{layers}"
            except ValueError:
                pass
    idx = 0
    while idx < len(tet_str) and tet_str[idx] == 'e':
        idx += 1
    if idx > 0:
        mant_str = tet_str[idx:]
        try:
            mant_val = mp.mpf(mant_str)
            if idx > 1:
                return f"E{comma_format(mant_val, decimals)}#{idx}"
            else:
                return f"{comma_format(addlayer(mant_val), decimals)}"
        except Exception:
            pass
    return tet_str

def format(tet, decimals=format_decimals):
    tet = correct(tet)
    if isinstance(tet, (int, float, mp.mpf)):
        return strip_trailing_zeros(comma_format(tet, decimals))
    tet_str = tet
    if tet_str.startswith("10^^"):
        height = mp.mpf(tet_str[4:])
        return f"F{strip_trailing_zeros(comma_format(height, 6))}"
    try:
        val = mp.mpf(tet_str)
        if abs(val) < mp.mpf('1e308'):
            return strip_trailing_zeros(comma_format(val, decimals))
    except Exception:
        pass
    if tet_str.startswith("(10^)^"):
        parts = tet_str.split(' ', 1)
        if len(parts) == 2:
            head, mant = parts
            try:
                layers = int(head[len("(10^)^"):])
                mant_val = mp.mpf(mant)
                if abs(mant_val - mp.mpf('1e10')) < mp.mpf('1e-5'):
                    return f"F{strip_trailing_zeros(comma_format(layers + 2, 6))}"
                elif mant_val < mp.mpf('10'):
                    mant_str = mp.nstr(mant_val, decimals+2)
                    return f"{mant_str}F{strip_trailing_zeros(comma_format(layers, 0))}"
                elif mant_val < mp.mpf('1e10'):
                    return f"{mp.nstr(mp.log10(mant_val), decimals+2)}F{strip_trailing_zeros(comma_format(layers + 1, 0))}"
                else:
                    return f"{mp.nstr(mp.log10(mp.log10(mant_val)), decimals+2)}F{strip_trailing_zeros(comma_format(layers + 2, 0))}"
            except ValueError:
                pass
    if tet_str.startswith('e'):
        idx = 0
        while idx < len(tet_str) and tet_str[idx] == 'e':
            idx += 1
        rest = tet_str[idx:]
        exp_pos = rest.rfind('e')
        if exp_pos > 0:
            mant_str = rest[:exp_pos]
            exp_str = rest[exp_pos+1:]
            try:
                mant_f = mp.mpf(mant_str)
                exp_i = int(exp_str)
                return f"{'e'*idx}{strip_trailing_zeros(comma_format(mant_f, decimals))}e{strip_trailing_zeros(comma_format(exp_i, 0))}"
            except Exception:
                pass
        try:
            mant = mp.mpf(rest)
            return f"{'e'*idx}{strip_trailing_zeros(comma_format(mant, decimals))}"
        except Exception:
            pass
    return tet_str

def power10_tower(tet, max_layers=max_layer, decimals=format_decimals):
    tet = correct(tet)
    s = slog(tet)
    if mp.isnan(s) or mp.isinf(s) or isinstance(s, str):
        return "NaN"
    if s > max_layers:
        return "10^^" + comma_format(s)
    height = int(mp.floor(s))
    frac = s - height
    if height <= 0:
        return frac
    mant = addlayer(frac, 2)
    expr = comma_format(mant, decimals)
    for _ in range(height - 1):
        expr = f"10^{expr}"
    return expr

def letter(s: str) -> str:
    s = correct(s)
    try:
        s = format_float_scientific(s)
    except:
        pass
    if gte(s, "(10^)^8 10000000000"):
        s = correct(s)
    if s.startswith("10^^") or s.startswith("(10^)^"):
        return format(s)

    if 'e' in s and not s.startswith('e') and not s.startswith("10^^") and not s.startswith("(10^)^"):
        parts = s.split('e', 1)
        if len(parts) == 2:
            try:
                mantissa = mp.mpf(parts[0])
                exponent_str = parts[1]

                if exponent_str.lstrip('-').lstrip('+').isdigit():
                    exponent_val = mp.mpf(exponent_str)
                else:
                    exponent_val = mp.mpf(exponent_str)

                leftover = mp.fmod(exponent_val, 3)  # high precision remainder
                group = mp.floor(exponent_val / 3) - 1  # high precision division

                new_mantissa = mantissa * mp.power(10, leftover)
                if new_mantissa >= mp.mpf('950'):
                    new_mantissa = mp.mpf(1)
                    group += 1

                if abs(new_mantissa - mp.nint(new_mantissa)) < mp.mpf('1e-5'):
                    formatted = str(int(mp.nint(new_mantissa)))
                else:
                    formatted = mp.nstr(new_mantissa, 3).rstrip('0').rstrip('.')

                if group < 0:
                    value = mantissa * mp.power(10, exponent_val)
                    if mp.floor(value) == value:
                        return str(int(value))
                    return mp.nstr(value, 3)
                elif group == 0:
                    return formatted + "K"
                elif group == 1:
                    return formatted + "M"
                elif group == 2:
                    return formatted + "B"
                else:
                    suffix = get_short_scale_suffix(int(group))
                    return formatted + suffix
            except Exception:
                pass

    k = 0
    while k < len(s) and s[k] == 'e':
        k += 1
    rest = s[k:]

    if k == 0:
        return s
    try:
        exponent_val = mp.mpf(rest)
        if exponent_val < 0:
            return "0"
    except Exception:
        return s

    if k == 1:
        if gt(exponent_val, suffix_max):
            return "e(" + str(letter(exponent_val)) + ")"

        if mp.floor(exponent_val) == exponent_val:
            leftover = mp.fmod(exponent_val, 3)
            group = mp.floor(exponent_val / 3) - 1
        else:
            leftover = mp.fmod(exponent_val, 3)
            group = mp.floor(exponent_val / 3) - 1

        if group < 0:
            value = mp.power(10, exponent_val)
            if mp.floor(value) == value:
                return str(int(value))
            else:
                return mp.nstr(value, 3)

        mantissa_val = mp.power(10, leftover)
        if mantissa_val >= mp.mpf('999.99'):
            mantissa_val = mp.mpf(1)
            group += 1

        if abs(mantissa_val - mp.nint(mantissa_val)) < mp.mpf('1e-5'):
            formatted = str(int(mp.nint(mantissa_val)))
        else:
            formatted = mp.nstr(mantissa_val, 3).rstrip('0').rstrip('.')

        if group == 0:
            return formatted + "K"
        elif group == 1:
            return formatted + "M"
        elif group == 2:
            return formatted + "B"
        else:
            suffix = get_short_scale_suffix(int(group))
            return formatted + suffix

    if k == 2:
        try:
            exponent_val = mp.mpf(rest)
            threshold = int(mp.ceil(mp.log10(suffix_max + 1))) if suffix_max > 0 else 0

            if exponent_val >= threshold:
                return 'e(' + letter("e" + rest) + ')'
            else:
                power_val = mp.power(10, exponent_val)
                try:
                    power_int = int(power_val)
                    group_index = (power_int - 3) // 3
                    suffix = get_short_scale_suffix(int(group_index))
                    return "10" + suffix
                except Exception:
                    return 'e(' + letter("e" + rest) + ')'
        except Exception:
            return 'e(' + letter("e" + rest) + ')'

    return fix_letter_output((k-2)*'e' + '(' + letter("ee" + rest) + ')')
def suffix_to_scientific(input_str: str) -> str:
    i = 0
    has_dot = False
    while i < len(input_str):
        c = input_str[i]
        if c in '0123456789':
            i += 1
        elif c == '.' and not has_dot:
            has_dot = True
            i += 1
        elif c == '-' and i == 0:
            i += 1
        else:
            break
    if i == 0:
        mantissa_val = mp.mpf('1.0')
        suffix_str = input_str
    else:
        mantissa_part = input_str[:i]
        suffix_str = input_str[i:]
        try:
            mantissa_val = mp.mpf(mantissa_part)
        except Exception:
            mantissa_val = mp.mpf('1.0')
            suffix_str = input_str

    additional_exponent = mp.mpf('0')
    if suffix_str:
        try:
            n = parse_suffix(suffix_str)
            additional_exponent = mp.mpf(3) * (mp.mpf(n) + 1)
        except Exception:
            additional_exponent = mp.mpf('0')

    if mantissa_val == 0:
        return "0"

    try:
        k = int(mp.floor(mp.log10(abs(mantissa_val))))
    except Exception:
        k = 0
    total_exponent = mp.mpf(k) + additional_exponent
    new_mantissa = mantissa_val / mp.power(10, k)

    if abs(new_mantissa - mp.nint(new_mantissa)) < mp.mpf('1e-5'):
        mantissa_output = str(int(mp.nint(new_mantissa)))
    else:
        formatted = mp.nstr(new_mantissa, 3)
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        mantissa_output = formatted

    if mantissa_output == "1":
        return "1e" + str(int(total_exponent))
    else:
        return mantissa_output + "1e" + str(int(total_exponent))

# Code helpers
def strip_trailing_zeros(num_str):
    if '.' in num_str:
        num_str = num_str.rstrip('0').rstrip('.')
    return num_str

def comma_format(number, decimals=format_decimals):
    try:
        num = mp.mpf(number)
    except Exception:
        return str(number)
    if abs(num) < mp.mpf('1e-3') or abs(num) >= mp.mpf('1e12'):
        if num == 0:
            return "0"
        sign = "-" if num < 0 else ""
        num_abs = abs(num)
        exp = int(mp.floor(mp.log10(num_abs)))
        mant = num_abs / mp.power(10, exp)
        mant_str = mp.nstr(mant, decimals + 2)
        if '.' in mant_str:
            parts = mant_str.split('.')
            frac = parts[1][:decimals]
            mant_str = parts[0] + (('.' + frac) if frac else '')
            mant_str = mant_str.rstrip('0').rstrip('.')
        return f"{sign}{mant_str}e{exp}"
    sign = "-" if num < 0 else ""
    num_abs = abs(num)
    int_part = int(mp.floor(num_abs))
    frac_part = num_abs - mp.mpf(int_part)
    frac_scaled = mp.nint(frac_part * mp.power(10, decimals))
    frac_str = str(int(frac_scaled)).rjust(decimals, '0') if decimals > 0 else ""
    int_str = str(int_part)
    int_parts = []
    while len(int_str) > 3:
        int_parts.insert(0, int_str[-3:])
        int_str = int_str[:-3]
    int_parts.insert(0, int_str)
    int_with_commas = ",".join(int_parts)
    if decimals > 0:
        return f"{sign}{int_with_commas}.{frac_str}"
    else:
        return f"{sign}{int_with_commas}"

def format_int_scientific(n: int, sig_digits: int = 16) -> str:
    n_m = mp.mpf(n)
    s = mp.nstr(n_m, sig_digits)
    if n_m == 0:
        return "0"
    exp = int(mp.floor(mp.log10(abs(n_m))))
    mant = n_m / mp.power(10, exp)
    mant_str = mp.nstr(mant, sig_digits).rstrip('0').rstrip('.')
    return f"{mant_str}e{exp}"

def format_float_scientific(x, sig_digits: int = 16) -> str:
    try:
        x_m = mp.mpf(x)
    except Exception:
        return str(x)
    if x_m <= 0 or mp.isinf(x_m) or mp.isnan(x_m):
        return str(x)
    exp = int(mp.floor(mp.log10(abs(x_m))))
    mant = x_m / mp.power(10, exp)
    mant_str = mp.nstr(mant, sig_digits).rstrip('0').rstrip('.')
    return f"{mant_str}e{exp}"

def correct(x):
    if not isinstance(x, (int, float, mp.mpf)):
        x = str(x).replace(",", "").strip()
    if isinstance(x, (int, float, mp.mpf)):
        return x
    x = str(x)
    if isinstance(x, str):
        x = x.strip()
        if "F" in x:
            f_index = x.find("F")
            if f_index > 0:
                before_f = x[:f_index]
                after_f = x[f_index+1:]
                try:
                    base = (before_f)
                    exp = (after_f)
                    x = "10^^" + str(exp + mp.log10(base))
                except Exception:
                    if x.startswith("F"):
                        x = tetr(10,x[1:])
            elif x.startswith("F"):
                try:
                    height = x[1:]
                    return tetr(10, height)
                except Exception:
                    pass
        if tetr(10, slog(x)) == "NaN":
            return suffix_to_scientific(x)
    return tetr(10, slog(x))
def correct2(x):
    if isinstance(x, (int, float, mp.mpf)):
        return str(x)
    x = x.strip()
    if "F" in x:
        f_index = x.find("F")
        before_f = x[:f_index]
        after_f = x[f_index+1:]
        try:
            base = mp.mpf(before_f) if before_f else mp.mpf(1)
            exp = mp.mpf(after_f)
        except Exception:
            return x
        if base == 1:
            return "10^^" + str(exp)
        else:
            val = exp + mp.log10(base)
            return "10^^" + str(val)
    if float(suffix_to_scientific(x)) > 1000:
        return suffix_to_scientific(x)
    
    return x
def fix_letter_output(s):
    cleaned = ''.join(c for c in s if c not in '()')
    e_count = 0
    i = 0
    while i < len(cleaned) and cleaned[i] == 'e':
        e_count += 1
        i += 1
    if i < len(cleaned) and cleaned[i].isdigit():
        prefix = 'e' * e_count
        suffix = cleaned[i:]
        return f"{prefix}({suffix})"
    else:
        return "Error: error formatting the short format value"

def get_short_scale_suffix(n: int) -> str:
    if n == 0:
        return ""
    if n < 1000:
        hundreds = n // 100
        tens = (n % 100) // 10
        units = n % 10
        return FirstOnes[units] + SecondOnes[tens] + ThirdOnes[hundreds]

    for i in range(len(MultOnes)-1, 0, -1):
        magnitude = 1000 ** i
        if n < magnitude:
            continue
        count = n // magnitude
        remainder = n % magnitude
        if count == 1:
            count_str = ""
        else:
            count_str = get_short_scale_suffix(count)

        rem_str = get_short_scale_suffix(remainder) if remainder > 0 else ""
        return count_str + MultOnes[i] + rem_str

    return ""

base_map = {}
for hundreds in range(0, 10):
    for tens in range(0, 10):
        for units in range(0, 10):
            s_str = FirstOnes[units] + SecondOnes[tens] + ThirdOnes[hundreds]
            num = hundreds * 100 + tens * 10 + units
            if s_str not in base_map:
                base_map[s_str] = num
base_map[""] = 0

mult_map = {}
for idx, s in enumerate(MultOnes):
    if s:
        if s in mult_map:
            if idx > mult_map[s]:
                mult_map[s] = idx
        else:
            mult_map[s] = idx

mult_strs_sorted = sorted([s for s in mult_map.keys() if s], key=len, reverse=True)

def parse_suffix(s: str) -> int:
    if s in base_map:
        return base_map[s]
    for mult_str in mult_strs_sorted:
        if s.endswith(mult_str):
            count_str = s[:-len(mult_str)]
            if mult_str == "":
                continue
            try:
                count_val = 1 if count_str == "" else parse_suffix(count_str)
                index_val = mult_map[mult_str]
                return count_val * (1000 ** index_val)
            except Exception:
                continue
    for i in range(0, len(s) + 1):
        for mult_str in mult_strs_sorted:
            if i + len(mult_str) <= len(s) and s.startswith(mult_str, i):
                count_str = s[:i]
                remainder_str = s[i + len(mult_str):]
                try:
                    count_val = 1 if count_str == "" else parse_suffix(count_str)
                    remainder_val = parse_suffix(remainder_str) if remainder_str else 0
                    index_val = mult_map[mult_str]
                    result = count_val * (1000 ** index_val) + remainder_val
                    return result
                except Exception:
                    continue
    return f"Unrecognized suffix: {s}"
bot.run(token, log_handler=handler, log_level=logging.DEBUG)


