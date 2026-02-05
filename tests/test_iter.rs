use proptest::prelude::*;
use tauri_plugin_llm::iter::*;

#[derive(Debug)]
pub struct TestConfig {
    pub random: usize,
    pub elements: usize,
    pub chunk_size: usize,
    pub expected_num_chunks: usize,
}

fn random_config() -> impl Strategy<Value = TestConfig> {
    (1usize..1000, 1usize..10000, 1usize..100).prop_map(|(random, elements, chunk_size)| {
        TestConfig {
            random,
            elements,
            chunk_size,
            expected_num_chunks: elements.div_ceil(chunk_size),
        }
    })
}

proptest! {

     #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))),
        .. ProptestConfig::default()
    })]
    #[test]
    fn test_iterable_chunks(config in random_config()) {
        struct ChunkedData {
            id: usize,
            data: Vec<u8>,
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let TestConfig { random, elements, chunk_size, expected_num_chunks } = config;

        let bytes = std::iter::repeat(1).collect::<Vec<u8>>();

        std::thread::spawn(move || {
            let _ = (&bytes).into_iter()
                .take(elements)
                .chunks(chunk_size)
                .enumerate()
                .try_for_each(|(id, chunk)| {
                    let data = chunk.cloned().collect();

                    if let Err(err) = tx.send(ChunkedData { id, data }) {
                        return Err(err);
                    }

                    Ok(())
                });
        });

        let mut chunks = vec![];

        while let Ok(chunk) = rx.recv() {
            chunks.push(chunk);
        }

        for (id, c) in chunks.iter().enumerate() {
            prop_assert!(c.data.len() <= chunk_size);
            prop_assert_eq!(id, c.id);
        }

        assert_eq!(chunks.len(), expected_num_chunks);
    }
}
