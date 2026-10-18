import asyncio, json, os, re, sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflparlay.parlay import Leg, evaluate_parlay, kelly_fraction
from playwright.async_api import async_playwright
import os
P=os.path.dirname(os.path.abspath(__file__))+'/'
D=json.load(open(P+'data.json')); G={g['id']:g for g in D['games']}
html=open(P+'nfl-parlay-desk.html').read()  # build first: python web/export_data.py
open(P+'wrapped.html','w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'+html+'</body></html>')

def leg(gid,side):
    g=G[gid]; home=side=='home'
    return Leg(g['home'] if home else g['away'], g['homeML'] if home else g['awayML'], g['homeProb'] if home else 1-g['homeProb'])
def num(s): return float(re.sub(r'[^0-9.\-]','',s.replace('−','-')))
fails=[]
def check(name,cond,detail=''):
    print(('PASS ' if cond else 'FAIL ')+name+(' '+detail if detail and not cond else ''))
    if not cond: fails.append(name)

async def slip(pg):
    rows=await pg.eval_on_selector_all('dl.out dt, dl.out dd','e=>e.map(x=>x.textContent)')
    return dict(zip(rows[::2],rows[1::2]))

async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        pg=await b.new_page(viewport={'width':1200,'height':900})
        errs=[]; pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file://'+P+'wrapped.html')
        check('12 games rendered', await pg.locator('.game').count()==12)
        check('24 side buttons', await pg.locator('.side').count()==24)
        check('example slip starts with 3 legs', await pg.locator('.leg').count()==3)

        async def build(sel, stake=None):
            if await pg.locator('#clear').count(): await pg.click('#clear')
            for gid,side in sel: await pg.click(f'.side[data-id="{gid}"][data-side="{side}"]')
            if stake is not None: await pg.fill('#stake',str(stake))
        async def compare(label, sel, stake=10):
            await build(sel, stake)
            s=await slip(pg); r=evaluate_parlay([leg(*x) for x in sel])
            k=kelly_fraction(r.model_win_prob,r.decimal_odds)
            got=dict(price=num(s['Combined price'].split('(')[0]), dec=num(s['Combined price'].split('(')[1]),
                     profit=num(s['Profit if it hits']), implied=num(s['Book implies a hit'])/100,
                     hit=num(s['Model hit chance'])/100, ev=num(s['Expected value']), kelly=num(s['Full-Kelly stake'].split('%')[0])/100)
            exp=dict(price=round(r.american_odds), dec=r.decimal_odds, profit=stake*r.payout_per_unit, implied=r.book_implied_prob,
                     hit=r.model_win_prob, ev=stake*r.ev_per_unit, kelly=k)
            tol=dict(price=1.01,dec=0.006,profit=0.006,implied=0.0006,hit=0.0006,ev=0.006,kelly=0.0006)
            bad={f:(got[f],round(exp[f],4)) for f in got if abs(got[f]-exp[f])>tol[f]}
            check(label, not bad, str(bad))
        await compare('2-leg BAL+BUF', [('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away')])
        await compare('3-leg longshot BAL+BUF+TEN $10', [('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away'),('2026_06_TEN_IND','away')])
        await compare('3-leg longshot, $25 stake', [('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away'),('2026_06_TEN_IND','away')],25)
        await compare('4-leg favorites', [('2026_06_ARI_LA','home'),('2026_06_WAS_SF','home'),('2026_06_LAC_KC','home'),('2026_06_NYJ_NE','home')])
        # interaction rules
        await build([('2026_06_BAL_CLE','away')])
        check('1 leg asks for more', 'Add one more game' in await pg.inner_text('#slipbody'))
        await pg.click('.side[data-id="2026_06_BAL_CLE"][data-side="home"]')
        check('picking other side swaps (one leg per game)', await pg.locator('.leg').count()==1 and 'CLE' in await pg.inner_text('.leg'))
        await pg.click('.side[data-id="2026_06_BAL_CLE"][data-side="home"]')
        check('tapping again removes leg', await pg.locator('.leg').count()==0)
        await build([('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away')])
        await pg.click('.leg .x >> nth=0')
        check('remove button works', await pg.locator('.leg').count()==1)
        await build([('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away')]); await pg.fill('#stake','0')
        s=await slip(pg); check('zero stake gives $0.00', num(s['Expected value'])==0 and num(s['Profit if it hits'])==0)
        # tabs
        await pg.click('#t-back'); check('backtest tab shows', await pg.locator('#p-back').is_visible() and not await pg.locator('#p-picks').is_visible())
        check('3 accuracy rows', await pg.locator('#acc tr').count()==4)
        check('5 chart bars', await pg.locator('#chart path').count()==5)
        await pg.hover('.hit >> nth=2'); await pg.wait_for_timeout(100)
        check('chart tooltip appears', await pg.locator('.tip').is_visible() and 'predicted' in await pg.inner_text('.tip'))
        check('no JS errors', not errs, str(errs))
        # phone width
        pg2=await b.new_page(viewport={'width':390,'height':800}); await pg2.goto('file://'+P+'wrapped.html')
        check('phone: no horizontal scroll', await pg2.evaluate('document.documentElement.scrollWidth')<=390)
        await pg2.click('#t-back'); check('phone: backtest no horizontal scroll', await pg2.evaluate('document.documentElement.scrollWidth')<=390)

        # ---- FanDuel price boxes ----
        pg3=await b.new_page(viewport={'width':1200,'height':900}); errs3=[]
        pg3.on('pageerror',lambda e:errs3.append(str(e)))
        await pg3.goto('file://'+P+'wrapped.html')
        check('24 price inputs', await pg3.locator('.pf input').count()==24)
        await pg3.click('#clear')
        for gid,side in [('2026_06_BAL_CLE','away'),('2026_06_BUF_LV','away')]: await pg3.click(f'.side[data-id="{gid}"][data-side="{side}"]')
        await pg3.fill('#fd-2026_06_BAL_CLE-away','-150')
        btn=await pg3.inner_text('.side[data-id="2026_06_BAL_CLE"][data-side="away"]')
        check('typed price shown on team button with FANDUEL tag', '150' in btn and 'FANDUEL' in btn)
        s=await slip(pg3); base=leg('2026_06_BAL_CLE','away')
        r=evaluate_parlay([Leg(base.label,-150,base.win_prob), leg('2026_06_BUF_LV','away')])
        check('slip uses typed -150 for BAL', abs(num(s['Combined price'].split('(')[1])-r.decimal_odds)<0.006 and abs(num(s['Expected value'])-10*r.ev_per_unit)<0.006, str(s))
        await pg3.fill('#fd-2026_06_BUF_LV-away','230')
        s=await slip(pg3); r2=evaluate_parlay([Leg(base.label,-150,base.win_prob), Leg('BUF',230,leg('2026_06_BUF_LV','away').win_prob)])
        check('unsigned 230 treated as +230', abs(num(s['Combined price'].split('(')[1])-r2.decimal_odds)<0.006)
        await pg3.fill('#fd-2026_06_BUF_LV-away','abc')
        check('invalid text flagged', await pg3.locator('#fd-2026_06_BUF_LV-away.bad').count()==1)
        s=await slip(pg3); r3=evaluate_parlay([Leg(base.label,-150,base.win_prob), leg('2026_06_BUF_LV','away')])
        check('invalid price falls back to data price', abs(num(s['Combined price'].split('(')[1])-r3.decimal_odds)<0.006)
        await pg3.fill('#fd-2026_06_BUF_LV-away','50')
        check('price under 100 rejected', await pg3.locator('#fd-2026_06_BUF_LV-away.bad').count()==1)
        await pg3.fill('#fd-2026_06_BUF_LV-away','\u2212245')
        check('unicode minus accepted', await pg3.locator('#fd-2026_06_BUF_LV-away.set').count()==1)
        await pg3.reload()
        v=await pg3.input_value('#fd-2026_06_BAL_CLE-away')
        print('INFO persistence after reload ->', repr(v))
        await pg3.click('#resetprices')
        check('reset clears typed prices', all(v=='' for v in await pg3.eval_on_selector_all('.pf input','e=>e.map(x=>x.value)')) and 'FANDUEL' not in await pg3.inner_text('#games'))
        check('no JS errors (price boxes)', not errs3, str(errs3))
        await b.close()
asyncio.run(main()); print('\nFAILURES:',fails or 'none')
