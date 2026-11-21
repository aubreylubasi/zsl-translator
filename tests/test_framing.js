// Run: node tests/test_framing.js
const assert=require('assert');
const Z=require(__dirname+'/../app/www/js/zsl.js');
function hand(cx,cy,s){const h=[];h[0]={x:cx,y:cy+s,z:0};for(let i=1;i<21;i++)h[i]={x:cx+(i%5-2)*s*0.3,y:cy-(i/21)*s,z:0};h[9]={x:cx,y:cy,z:0};return h;}
const a=9/16, pose=(lx,ly,rx,ry,v=0.99)=>{const p=[];for(let i=0;i<33;i++)p[i]={x:.5,y:.5,z:0,visibility:v};p[11]={x:lx,y:ly,z:0,visibility:v};p[12]={x:rx,y:ry,z:0,visibility:v};return p;};
const cases={
 'normal signer, hand by chest':[[hand(.45,.55,.06)],pose(.7,.7,.3,.7)],
 'your screenshot: shoulders on the palm':[[hand(.5,.45,.12)],pose(.58,.47,.42,.47)],
 'no body found':[[hand(.5,.45,.12)],null],
 'shoulders hidden (low visibility)':[[hand(.45,.55,.06)],pose(.7,.7,.3,.7,0.2)],
 'shoulders below the picture':[[hand(.45,.55,.06)],pose(.7,1.1,.3,1.1)],
 'hand right at the camera':[[hand(.5,.4,.3)],pose(.8,.85,.2,.85)],
};
const want={'normal signer, hand by chest':null};
for(const [k,[h,p]] of Object.entries(cases)){const r=Z.framingIssue(h,p,a);console.log((r||'OK').padEnd(10),k);
 if(k in want) assert.strictEqual(r,null,k); else assert.ok(r,k+' should be caught');}
const t=new Z.Typist(['A','B']);t.text='AB';t.pause();assert.strictEqual(t.text,'AB');assert.strictEqual(t.guess,null);console.log('all framing tests passed');
