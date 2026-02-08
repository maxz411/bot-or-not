/**
 * Data helpers for reading dataset JSON files.
 */

export type Post = {
  text: string;
  created_at: string;
  id: string;
  author_id: string;
  lang: string;
};

export type User = {
  id: string;
  tweet_count: number;
  z_score: number;
  username: string;
  name: string;
  description: string;
  location: string;
};

export type Dataset = {
  id: number;
  lang: string;
  metadata: Record<string, unknown>;
  posts: Post[];
  users: User[];
};

/** Load and parse a dataset JSON file. */
export async function loadDataset(path: string): Promise<Dataset> {
  const file = Bun.file(path);
  return file.json() as Promise<Dataset>;
}

/** Return all users across multiple dataset files (deduped by ID). */
export async function getAllUsers(
  datasetPaths: string[],
): Promise<User[]> {
  const datasets = await Promise.all(datasetPaths.map(loadDataset));
  const seen = new Set<string>();
  const users: User[] = [];
  for (const ds of datasets) {
    for (const u of ds.users) {
      if (!seen.has(u.id)) {
        seen.add(u.id);
        users.push(u);
      }
    }
  }
  return users;
}

/** Find a user by ID or username across loaded datasets. */
function findUser(datasets: Dataset[], idOrUsername: string): User | undefined {
  for (const ds of datasets) {
    const user = ds.users.find(
      (u) => u.id === idOrUsername || u.username === idOrUsername,
    );
    if (user) return user;
  }
  return undefined;
}

/** Return user metadata for a given user ID or username across multiple dataset files. */
export async function getUserMetadata(
  idOrUsername: string,
  datasetPaths: string[],
): Promise<User | undefined> {
  const datasets = await Promise.all(datasetPaths.map(loadDataset));
  return findUser(datasets, idOrUsername);
}

/** Return all posts for a given user ID or username across multiple dataset files. */
export async function getPostsByUser(
  idOrUsername: string,
  datasetPaths: string[],
): Promise<Post[]> {
  const datasets = await Promise.all(datasetPaths.map(loadDataset));
  return getPostsByUserFromLoaded(idOrUsername, datasets);
}

/** Same as getPostsByUser but works with already-loaded datasets (avoids re-parsing). */
export function getPostsByUserFromLoaded(
  idOrUsername: string,
  datasets: Dataset[],
): Post[] {
  const user = findUser(datasets, idOrUsername);
  if (!user) return [];
  return datasets.flatMap((ds) =>
    ds.posts.filter((p) => p.author_id === user.id),
  );
}

/** Load multiple datasets once and return them. */
export async function loadDatasets(paths: string[]): Promise<Dataset[]> {
  return Promise.all(paths.map(loadDataset));
}
