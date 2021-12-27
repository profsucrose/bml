import { App, LogLevel } from '@slack/bolt'
import axios from 'axios'
import * as fs from 'fs'
import { spawn } from 'child_process'
import { unescape } from 'html-escaper'

const { TOKEN, SIGNING_SECRET } = JSON.parse(fs.readFileSync('auth.json').toString())

const app = new App({
    token: TOKEN,
    signingSecret: SIGNING_SECRET,
    logLevel: LogLevel.DEBUG
})

const CHANNEL_ID = 'C02RJRTK8LV'
const TMP_IMAGE_DOWNLOAD = './tmp/tmp.png'
const TMP_BML_OUTPUT = './tmp/out.png'
const TMP_SCRIPT_DOWNLOAD = './tmp/script.bml'

const PORT = 3004

async function downloadImage(url: string) {
    const response = await axios({
        url,
        method: 'get',
        responseType: 'stream',
        headers: {
            Authorization: `Bearer ${TOKEN}`
        }
    })

    response.data.pipe(fs.createWriteStream(TMP_IMAGE_DOWNLOAD))
}

async function runBML(): Promise<string> {
    const bml = spawn('../bml/target/release/bml', ['process', TMP_SCRIPT_DOWNLOAD, TMP_IMAGE_DOWNLOAD, '1', TMP_BML_OUTPUT])

    return new Promise((resolve, reject) => {
        let output = ''

        bml.stdout.on('data', (data: Buffer) => { output += data.toString() })
        bml.stderr.on('data', (data: Buffer) => { output += data.toString() })

        bml.on('exit', (code, signal) => {
            console.log('Process ended', code, signal)
            if (code == 0) resolve(output)
            reject(output)
        })
    })
}

async function lastImage(): Promise<string> {
    let history = await app.client.conversations.history({
        channel: CHANNEL_ID
    })

    for (;;) {
        for (let msg of history.messages || []) {
            if (msg.files && msg.files[0].filetype == 'png') {
                return msg.files[0].url_private_download!
            }
        }

        if (history.response_metadata?.next_cursor) {
            history = await app.client.conversations.history({
                channel: CHANNEL_ID,
                cursor: history.response_metadata.next_cursor!
            })
            continue
        }

        // no more pages left
        break
    }

    throw new Error('couldn\'t find image in channel history')
}

app.message(async ({ message, say }) => {
    const url = await lastImage()

    console.log('DOWNLOAD URL', url)

    await downloadImage(url)

    const groups = <string>(message['text']).match(/```((.|\n)*?)```/)

    let script = groups && unescape(groups[1])

    console.log(script)

    if (!script) {
        return
    }

    console.log(`Script: '${script}'`)

    await new Promise((resolve, reject) => {
        fs.writeFile(TMP_SCRIPT_DOWNLOAD, script, (err) => {
            if (err) reject(err)
            resolve('')
        })
    })

    try {
        await runBML()

        // send images
        app.client.files.upload({
            channels: CHANNEL_ID,
            initial_comment: 'Manipulated image',
            file: fs.createReadStream(TMP_BML_OUTPUT)
        })
    } catch (err) {
        say(<string>`Error: \`\`\`${err}\`\`\``)
    }
})

;(async () => {
    await app.start(PORT)
    console.log(`The Slack app has started!`)
})()