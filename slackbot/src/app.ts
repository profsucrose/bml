import { App, LogLevel } from '@slack/bolt'
import axios from 'axios'
import fs, { createReadStream } from 'fs'
import { spawn } from 'child_process'
import * as dotenv from 'dotenv'
import { unescape } from 'html-escaper'
dotenv.config()

const app = new App({
    token: process.env.TOKEN,
    signingSecret: process.env.SIGNING_SECRET,
    logLevel: LogLevel.DEBUG
})

const CHANNEL_ID = 'C02RJRTK8LV'
const TMP_IMAGE_DOWNLOAD = 'tmp.png'
const TMP_BML_OUTPUT = 'out.png'
const TMP_SCRIPT_DOWNLOAD = 'script.bml'

async function downloadImage(url: string) {
    const response = await axios({
        url,
        method: 'get',
        responseType: 'stream',
        headers: {
            Authorization: `Bearer ${process.env.TOKEN}`
        }
    })

    response.data.pipe(fs.createWriteStream(TMP_IMAGE_DOWNLOAD))
}

async function runBML(): Promise<string> {
    const bml = spawn('./bml/target/release/bml', [TMP_SCRIPT_DOWNLOAD, TMP_IMAGE_DOWNLOAD])

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
    await downloadImage(url)

    const groups = <string>(message.text).match(/```((.|\n)*?)```/)

    let script = groups && groups[1]

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
            file: createReadStream(TMP_BML_OUTPUT)
        })
    } catch (err) {
        say(<string>`Error: \`\`\`${err}\`\`\``)
    }
})

;(async () => {
    await app.start(3000)
    console.log(`The Slack app has started!`)
})()