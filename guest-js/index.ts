import { invoke } from '@tauri-apps/api/core'

// export async function ping(value: string): Promise<string | null> {
//   return await invoke<{ value?: string }>('plugin:tauri-plugin-llm|ping', {
//     payload: {
//       value,
//     },
//   }).then((r) => (r.value ? r.value : null));
// }



export async function send_message(value: string): Promise<string | null> {
  return await invoke<{ value?: string }>('plugin:llm|send_message', {
    payload: {
      value,
    },
  }).then((r) => (r.value ? r.value : null));
}

export async function check_status(value: string): Promise<string | null> {
  return await invoke<{ value?: string }>('plugin:llm|check_status', {
    payload: {
      value,
    },
  }).then((r) => (r.value ? r.value : null));
}
