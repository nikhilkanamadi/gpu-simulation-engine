declare module "sql.js" {
  const initSqlJs: (opts: { locateFile?: (file: string) => string }) => Promise<any>;
  export default initSqlJs;
}

