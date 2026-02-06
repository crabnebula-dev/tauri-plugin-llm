/// Extension trait that adds chunking capability to any iterator.
///
/// This trait provides the [`chunks`](IntoIterChunks::chunks) method which groups
/// iterator items into fixed-size chunks, yielding each chunk as soon as it's
/// complete. This enables streaming behavior where chunks are available
/// incrementally rather than waiting for the entire iterator to be consumed.
///
/// # Example
///
/// ```
/// use tauri_plugin_llm::iter::*;
///
/// // With owned values, use into_iter()
/// let numbers: Vec<Vec<i32>> = (1..=10)
///     .chunks(3)
///     .map(|chunk| chunk.into_iter().collect())
///     .collect();
///
/// assert_eq!(numbers, vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
///     vec![10],
/// ]);
///
/// // With references, use cloned() to get owned values
/// let data = vec![1, 2, 3, 4, 5];
/// let chunks: Vec<Vec<i32>> = data.iter()
///     .chunks(2)
///     .map(|chunk| chunk.cloned().collect())
///     .collect();
///
/// assert_eq!(chunks, vec![vec![1, 2], vec![3, 4], vec![5]]);
/// ```
pub trait IntoIterChunks: Iterator + Sized {
    /// Groups iterator items into chunks of the specified size.
    ///
    /// Returns a [`Chunks`] iterator that yields [`Chunk`] items, each containing
    /// up to `size` elements. The final chunk may contain fewer elements if the
    /// iterator length is not evenly divisible by `size`.
    ///
    /// Chunks are yielded as soon as they are filled, making this suitable for
    /// streaming scenarios where items are produced slowly.
    ///
    /// # Panics
    ///
    /// Does not panic if `size` is 0, but will yield empty chunks indefinitely.
    /// Callers should ensure `size >= 1`.
    fn chunks(self, size: usize) -> Chunks<Self> {
        Chunks::new(self, size)
    }
}

impl<I: Iterator> IntoIterChunks for I {}

/// An iterator adapter that groups items into fixed-size chunks.
///
/// Created by the [`IntoIterChunks::chunks`] method. See its documentation for more.
pub struct Chunks<I: Iterator> {
    iter: I,
    size: usize,
}

impl<I: Iterator> Chunks<I> {
    fn new(iter: I, size: usize) -> Self {
        Self { iter, size }
    }
}

impl<I: Iterator> Iterator for Chunks<I> {
    type Item = Chunk<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let items: Vec<_> = (&mut self.iter).take(self.size).collect();
        if items.is_empty() {
            None
        } else {
            Some(Chunk { items })
        }
    }
}

/// A container holding a single chunk of items from a [`Chunks`] iterator.
///
/// Use [`cloned`](Chunk::cloned) to iterate over the contained items.
pub struct Chunk<T> {
    items: Vec<T>,
}

impl<T> IntoIterator for Chunk<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, T: Clone> Chunk<&'a T> {
    /// Returns an iterator that clones each referenced item.
    ///
    /// # Example
    ///
    /// ```
    /// use tauri_plugin_llm::iter::IntoIterChunks;
    ///
    /// let data = vec![1, 2, 3];
    /// let chunk = data.iter().chunks(3).next().unwrap();
    /// let items: Vec<i32> = chunk.cloned().collect();
    /// assert_eq!(items, vec![1, 2, 3]);
    /// ```
    pub fn cloned(self) -> impl Iterator<Item = T> + use<'a, T> {
        self.items.into_iter().cloned()
    }
}
