import {createRequire} from 'node:module';
const require=createRequire(new URL('../../online_web/package.json',import.meta.url));
const {chromium}=require('playwright');

export function launchBrowser(){
  const options={headless:true,args:['--disable-dev-shm-usage']};
  if(process.env.LIGHTMEM_BROWSER_EXECUTABLE)options.executablePath=process.env.LIGHTMEM_BROWSER_EXECUTABLE;
  return chromium.launch(options);
}
